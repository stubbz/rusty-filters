
extern crate symbolic_polynomials;

use self::symbolic_polynomials::{SymPolynomial, SymMonomial};

pub struct BilinearTransform {
    transform: Vec<Vec<f32>>
}

impl BilinearTransform {
    #[inline]
    pub fn new(order: i32) -> Self {
        let transform = calculate_bilinear_transform_matrix(order);
        BilinearTransform {
            transform: transform
        }
    }

    #[inline]
    pub fn order(&self) -> usize {
        (self.transform.len() - 1)
    }

    #[inline]
    pub fn transform(&self, input_vector: &[f32], k: f32) -> Vec<f32>{
        let num_coefficients = self.transform.len();
        let mut output_vector = Vec::with_capacity(num_coefficients);

        for i in 0..num_coefficients{
            let mut output = 0.0;
            for j in 0..num_coefficients{
                let k_pow = (num_coefficients - j - 1) as i32;
                let value = k.powi(k_pow) * input_vector[j] * self.transform[j][i];
                output = output + value;
            }
            output_vector.push(output);
        }
        output_vector
    }
}

#[inline]
fn calculate_bilinear_transform_matrix(order: i32) -> Vec<Vec<f32>>{//todo make fixed size/generic
    //uses polynomials where 'x' = z^-1
    fn create_polynomial(coefficients: Vec<i32>) -> SymPolynomial{
        let mut polynomial = SymPolynomial::new(1);
        for (power,coefficient) in coefficients.iter().enumerate() {
            let mut monomial = SymMonomial::new(1);
            monomial.powers[0] = power as usize;
            monomial.coefficient = *coefficient;
            polynomial.monomials.push(monomial);
        }
        polynomial.simplify();
        polynomial
    }


    let num_coefficients = (order + 1) as usize;
    let mut result = Vec::with_capacity(num_coefficients);
    for i in 0..num_coefficients{

        let mut polynomial = create_polynomial(vec![1]);

        let num_negative = (num_coefficients - i - 1) as usize;
        for _ in 0..num_negative{
            let negative = create_polynomial(vec![1, -1]);
            polynomial = &polynomial * &negative;
        }
        let num_positive = i;
        for _ in 0..num_positive{
            let positive = create_polynomial(vec![1, 1]);
            polynomial = &polynomial * &positive;
        }

        let mut coefficients = Vec::with_capacity(num_coefficients);
        for order in 0..num_coefficients{
            let mut added = false;
            for monomial in &polynomial.monomials {
                if monomial.powers[0] == order{
                    coefficients.push(monomial.coefficient as f32);
                    added = true;
                    break;
                }
            }
            if !added {
                coefficients.push(0.0);
            }
        }
        result.push(coefficients);
    }
    result
}

#[test]
fn bilinear_transform_matrix_test_2nd_order() {
    let result = calculate_bilinear_transform_matrix(2);
    let expected = vec![
        vec![1.0, -2.0,  1.0],
        vec![1.0,  0.0, -1.0],
        vec![1.0,  2.0,  1.0]
    ];
    assert_eq!(result, expected);
}

#[test]
fn bilinear_transform_matrix_test_4th_order() {
    let result = calculate_bilinear_transform_matrix(4);
    let expected = vec![
        vec![1.0, -4.0,  6.0, -4.0,  1.0],
        vec![1.0, -2.0,  0.0,  2.0, -1.0],
        vec![1.0,  0.0, -2.0,  0.0,  1.0],
        vec![1.0,  2.0,  0.0, -2.0, -1.0],
        vec![1.0,  4.0,  6.0,  4.0,  1.0]
    ];
    assert_eq!(result, expected);
}


#[test]
fn bilinear_transform_test_2nd_order() {
    let transform = BilinearTransform::new(2);
    let result = transform.transform(&[1.0, 1.0, 1.0], 1.0);
    let expected = [3.0, 0.0, 1.0];
    assert_eq!(result, expected);
}

#[test]
fn bilinear_transform_test_2nd_order_test_k() {
    let transform = BilinearTransform::new(2);
    let result = transform.transform(&[1.0, 1.0, 1.0], 0.5);
    let expected = [1.75, 1.5, 0.75];
    assert_eq!(result, expected);
}
