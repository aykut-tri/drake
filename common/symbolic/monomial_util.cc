#include "drake/common/symbolic/monomial_util.h"

#include "drake/common/drake_assert.h"

namespace drake {
namespace symbolic {
Eigen::Matrix<Monomial, Eigen::Dynamic, 1> MonomialBasis(const Variables& vars,
                                                         const int degree) {
  return internal::ComputeMonomialBasis<Eigen::Dynamic>(
      vars, degree, internal::DegreeType::kAny);
}

Eigen::Matrix<Monomial, Eigen::Dynamic, 1> EvenDegreeMonomialBasis(
    const Variables& vars, int degree) {
  return internal::ComputeMonomialBasis<Eigen::Dynamic>(
      vars, degree, internal::DegreeType::kEven);
}

Eigen::Matrix<Monomial, Eigen::Dynamic, 1> OddDegreeMonomialBasis(
    const Variables& vars, int degree) {
  return internal::ComputeMonomialBasis<Eigen::Dynamic>(
      vars, degree, internal::DegreeType::kOdd);
}
}  // namespace symbolic
}  // namespace drake
