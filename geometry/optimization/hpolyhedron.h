#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hyperellipsoid.h"

namespace drake {
namespace geometry {
namespace optimization {

/** Implements a polyhedral convex set using the half-space representation:
`{x| A x ≤ b}`.  Note: This set may be unbounded.
@ingroup geometry_optimization
*/
class HPolyhedron final : public ConvexSet {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(HPolyhedron)

  /** Constructs the polyhedron.
  @pre A.rows() == b.size().
  */
  HPolyhedron(const Eigen::Ref<const Eigen::MatrixXd>& A,
              const Eigen::Ref<const Eigen::VectorXd>& b);

  /** Constructs a new HPolyhedron from a SceneGraph geometry and pose in the
  @p reference_frame frame, obtained via the QueryObject.  If @p reference_frame
  frame is std::nullopt, then it will be expressed in the world frame.

  @throws std::exception the geometry is not a convex polytope. */
  HPolyhedron(const QueryObject<double>& query_object, GeometryId geometry_id,
              std::optional<FrameId> reference_frame = std::nullopt);
  // TODO(russt): Add a method/constructor that would create the geometry using
  // SceneGraph's AABB or OBB representation (for arbitrary objects) pending
  // #15121.

  ~HPolyhedron() final;

  /** Returns the half-space representation matrix A. */
  const Eigen::MatrixXd& A() const { return A_; }

  /** Returns the half-space representation vector b. */
  const Eigen::VectorXd& b() const { return b_; }

  /** Returns true iff the set is bounded, e.g. there exists an element-wise
  finite lower and upper bound for the set.  For HPolyhedron, while there are
  some fast checks to confirm a set is unbounded, confirming boundedness
  requires solving a linear program (based on Stiemke’s theorem of
  alternatives). */
  using ConvexSet::IsBounded;

  /** Returns true iff this HPolyhedron is entirely contained in the HPolyhedron
  other. This is done by checking whether every inequality in @p other is
  redundant when added to this. */
  bool ContainedIn(const HPolyhedron& other) const;

  /** Constructs the intersection of two HPolyhedron by adding the rows of
  inequalities from @p other. If @p check_for_redundancy is true
  then only adds the rows of @p other to this HPolyhedron if the inequality
  is not implied by the inequalities from this HPolyhedron. */
  HPolyhedron Intersection(const HPolyhedron& other,
                           bool check_for_redundancy = false) const;

  /** Reduces some (not necessarily all) redundant inequalities in the
  HPolyhedron.  This is not guaranteed to give the minimal representation of
  the polyhedron but is a relatively fast way to reduce the number of
  inequalities. */
  HPolyhedron ReduceInequalities() const;

  /** Solves a semi-definite program to compute the inscribed ellipsoid.
  From Section 8.4.2 in Boyd and Vandenberghe, 2004, we solve
  @verbatim
  max_{C,d} log det (C)
        s.t. |aᵢC|₂ ≤ bᵢ - aᵢd, ∀i
            C ≽ 0
  @endverbatim
  where aᵢ and bᵢ denote the ith row.  This defines the ellipsoid
  E = { Cx + d | |x|₂ ≤ 1}.

  @pre the HPolyhedron is bounded.
  @throws std::exception if the solver fails to solve the problem.
  */
  Hyperellipsoid MaximumVolumeInscribedEllipsoid() const;

  /** Solves a linear program to compute the center of the largest inscribed
  ball in the polyhedron.  This is often the recommended way to find some
  interior point of the set, for example, as a step towards computing the convex
  hull or a vertex-representation of the set.

  Note that the Chebyshev center is not necessarily unique, and may not conform
  to the point that one might consider the "visual center" of the set.  For
  example, for a long thin rectangle, any point in the center line segment
  illustrated below would be a valid center point.  The solver may return
  any point on that line segment.
  @verbatim
    ┌──────────────────────────────────┐
    │                                  │
    │   ────────────────────────────   │
    │                                  │
    └──────────────────────────────────┘
  @endverbatim
  To find the visual center, consider using the more expensive
  MaximumVolumeInscribedEllipsoid() method, and then taking the center of the
  returned Hyperellipsoid.

  @throws std::exception if the solver fails to solve the problem.
  */
  Eigen::VectorXd ChebyshevCenter() const;

  /** Returns the Cartesian product of `this` and `other`. */
  HPolyhedron CartesianProduct(const HPolyhedron& other) const;

  /** Returns the `n`-ary Cartesian power of `this`.
  The n-ary Cartesian power of a set H is the set H ⨉ H ⨉ ... ⨉ H, where H is
  repeated n times. */
  HPolyhedron CartesianPower(int n) const;

  /** Returns the Pontryagin (Minkowski) Difference of `this` and `other`.
  This is the set A ⊖ B = { a|a+ B ⊆ A }. The result is an HPolyhedron with the
  same number of inequalities as A. Requires that `this` and `other` both
  be bounded and have the same ambient dimension. This method may throw a
  runtime error if `this` or `other` are ill-conditioned. */
  HPolyhedron PontryaginDifference(const HPolyhedron& other) const;

  /** Draw an (approximately) uniform sample from the set using the hit and run
  Markov-chain Monte-Carlo strategy described at
  https://mathoverflow.net/a/162327 and the cited paper.

  To generate many samples, pass the output of one iteration in as the @p
  previous_sample to the next; in this case the distribution of samples will
  converge to the true uniform distribution in total variation at a geometric
  rate.  If @p previous_sample is not set, then the ChebyshevCenter() will be
  used to seed the algorithm.

  @throws std::exception if previous_sample is not in the set.
  */
  Eigen::VectorXd UniformSample(
      RandomGenerator* generator,
      const Eigen::Ref<Eigen::VectorXd>& previous_sample) const;

  /** Variant of UniformSample that uses the ChebyshevCenter() as the
  previous_sample as a feasible point to start the Markov chain sampling. */
  Eigen::VectorXd UniformSample(RandomGenerator* generator) const;

  /** Constructs a polyhedron as an axis-aligned box from the lower and upper
  corners. */
  static HPolyhedron MakeBox(const Eigen::Ref<const Eigen::VectorXd>& lb,
                             const Eigen::Ref<const Eigen::VectorXd>& ub);

  /** Constructs the L∞-norm unit box in @p dim dimensions, {x | |x|∞ <= 1 }.
  This is an axis-aligned box, centered at the origin, with edge length 2. */
  static HPolyhedron MakeUnitBox(int dim);

  /** Constructs the L1-norm unit ball in @p dim dimensions, {x | |x|₁ <= 1 }.
  This set is also known as the crosspolytope and is described by the 2ᵈⁱᵐ
  signed unit vectors. */
  static HPolyhedron MakeL1Ball(int dim);

 private:
  HPolyhedron DoIntersectionNoChecks(const HPolyhedron& other) const;

  HPolyhedron DoIntersectionWithChecks(const HPolyhedron& other) const;

  bool DoIsBounded() const final;

  bool DoPointInSet(const Eigen::Ref<const Eigen::VectorXd>& x,
                    double tol) const final;

  void DoAddPointInSetConstraints(
      solvers::MathematicalProgram* prog,
      const Eigen::Ref<const solvers::VectorXDecisionVariable>& vars)
      const final;

  std::vector<solvers::Binding<solvers::Constraint>>
  DoAddPointInNonnegativeScalingConstraints(
      solvers::MathematicalProgram* prog,
      const Eigen::Ref<const solvers::VectorXDecisionVariable>& x,
      const symbolic::Variable& t) const final;

  std::vector<solvers::Binding<solvers::Constraint>>
  DoAddPointInNonnegativeScalingConstraints(
      solvers::MathematicalProgram* prog,
      const Eigen::Ref<const Eigen::MatrixXd>& A_x,
      const Eigen::Ref<const Eigen::VectorXd>& b_x,
      const Eigen::Ref<const Eigen::VectorXd>& c, double d,
      const Eigen::Ref<const solvers::VectorXDecisionVariable>& x,
      const Eigen::Ref<const solvers::VectorXDecisionVariable>& t) const final;

  // TODO(russt): Implement DoToShapeWithPose.  Currently we don't have a Shape
  // that can consume this output.  The obvious candidate is Convex, that class
  // currently only stores the filename of an .obj file, and I shouldn't need
  // to go to disk and back to get this geometry into ProximityEngine nor
  // MeshcatVisualizer.
  //
  // When we do implement this and also have the AHPolyhedron class, we should
  // add recommendation here that: "If ambient_dimension() != 3, then consider
  // using the AH polytope representation to project it to 3D."
  std::pair<std::unique_ptr<Shape>, math::RigidTransformd> DoToShapeWithPose()
      const final;

  // Implement support shapes for the ShapeReifier interface.
  using ShapeReifier::ImplementGeometry;
  void ImplementGeometry(const HalfSpace&, void* data) final;
  void ImplementGeometry(const Box& box, void* data) final;
  // TODO(russt): Support ImplementGeometry(const Convex& convex, ...), but
  // currently it would require e.g. digging ReadObjForConvex out of
  // proximity_engine.cc.

  Eigen::MatrixXd A_{};
  Eigen::VectorXd b_{};
};

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
