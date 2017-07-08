
use std::collections::HashMap;
use std::f32;
use std::fmt::Debug;
use std::cmp::Ordering;
use std::fmt;
extern crate num;
use std::ops::{Add,Mul,Neg};
use self::num::traits::{Zero,One};
use self::num::complex::Complex;

#[derive(Debug)]
pub struct Polynomial<TCoefficient> {
    map: HashMap<i32, TCoefficient>
}

pub type ComplexPolynomial<T> = Polynomial<Complex<T>>;

impl<TCoefficient: Copy + One + Zero +
Neg<Output=TCoefficient>
+ Mul<TCoefficient,Output=TCoefficient>
+ Add<TCoefficient,Output=TCoefficient>
> Polynomial<TCoefficient>{
    pub fn new() -> Self{
        Polynomial{
            map: HashMap::new()
        }
    }

    pub fn set_coefficient(&mut self, order: i32, coeff: TCoefficient){
        self.map.insert(order, coeff);
    }

    pub fn get_orders(&self) -> Vec<i32> {
        let mut keys = self.map.keys().map(|order| *order).collect::<Vec<_>>();
        keys.sort_by(|a,b| a.cmp(b));
        keys
    }

    pub fn get_coefficient(&self, order: i32) -> TCoefficient {
        match self.map.get(&order) {
            Some(coeff) => *coeff,
            None => TCoefficient::zero()
        }
    }
    pub fn from_root(root: TCoefficient) -> Self{
        let mut polynomial = Polynomial::new();
        polynomial.set_coefficient(1, TCoefficient::one());
        polynomial.set_coefficient(0, - root);
        polynomial
    }

    #[inline]
    pub fn to_vector(&self) -> Vec<TCoefficient>{
        let mut res = Vec::new();
        let mut max_order_opt = self.map.keys().max();

        match max_order_opt {
            Some(max_order) => {
                for order in 0..(max_order + 1){
                    res.push(self.get_coefficient(order));
                }
                res
            }
            None => res
        }
    }
}


impl<TCoefficient: Copy + One + Zero +
Neg<Output=TCoefficient>
+ Mul<TCoefficient,Output=TCoefficient>
+ Add<TCoefficient,Output=TCoefficient>
+ Debug
+ fmt::Display> fmt::Display for Polynomial<TCoefficient> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        for (order,coeff) in self.to_vector().iter().enumerate(){
            writeln!(f, "{} * x^{}", coeff, order);
        }



        Ok(())
    }
}

impl<T: Copy + One + Zero + Mul<T, Output=T> + Neg<Output=T> + Add<T, Output=T> + Debug>
    Polynomial<T>
    {

    #[inline]
    fn multiply_internal(&self, other: &Polynomial<T>) -> Polynomial<T>{
        let mut result = Polynomial::new();
        for (lhs_order, lhs_coeff) in &self.map {
            for(rhs_order, rhs_coeff) in &other.map {
                let order = (*lhs_order) + (*rhs_order);
                let coeff = result.get_coefficient(order) + (*lhs_coeff) * (*rhs_coeff);
                result.map.insert(order, coeff);
            }
        }
        result
    }

    pub fn multiply_polynomials(polynomials: &[Polynomial<T>]) -> Polynomial<T> {
        //todo this might be faster if some analysis is used in order to not create new polynomials for each iter of the fold
        //alternatively a single mutable polynomial might work good as well
        let mut one = Polynomial::new();
        one.set_coefficient(0, T::one());
        let multiplied_polynomial = polynomials.iter().fold(one, |acc, polynomial| polynomial.multiply_internal(&acc));
        multiplied_polynomial
    }
}

impl<'a, 'b, TCoefficient: Copy + Zero + One
+ Mul<TCoefficient, Output=TCoefficient>
+ Add<TCoefficient, Output=TCoefficient>
+ Neg<Output=TCoefficient>
+ Debug>
Mul<&'b Polynomial<TCoefficient>>
for &'a Polynomial<TCoefficient> {
    type Output = Polynomial<TCoefficient>;

    #[inline]
    fn mul(self, other: &'b Polynomial<TCoefficient>) -> Polynomial<TCoefficient> {
        self.multiply_internal(other)
    }
}
