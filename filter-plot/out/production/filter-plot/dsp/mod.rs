pub mod bilinear_transform;

extern crate num;
use self::num::complex::Complex;

pub trait HasMagnitude<T>{
    fn magnitude(&self) -> f32;
}

impl HasMagnitude<f32> for Complex<f32> {
    fn magnitude(&self) -> f32 {
        self.re.hypot(self.im)
    }
}

#[test]
fn dft_test() {
    let res = dft(&vec![1.0,2.0,4.0,2.0,1.0], 5);
    let supposedToBe = vec![
        Complex::new(10.0, 0.0),
        Complex::new(-2.92705, -2.12663),
        Complex::new(0.42705, 1.31433),
        Complex::new(0.42705, -1.31433),
        Complex::new(-2.92705, 2.12663)
    ];

    for (r,s) in res.iter().zip(supposedToBe.iter()){
        let diff = r-s;
        println!("res={}", r);
        println!("expected={}", s);
        println!("diff={}", diff);
        if(diff.re.abs() > 0.01 || diff.im.abs() > 0.01){

            panic!("test failed");

        }
    }
}

#[inline]
pub fn dft(signal: &[f32], sample_rate: usize) -> Vec<Complex<f32>> {
    use std::f32::consts;


    let mut dft = Vec::with_capacity(sample_rate);

    let signal = signal.iter().map(|sample| Complex::new(*sample, 0.0)).collect::<Vec<Complex<f32>>>();
    let n = sample_rate as f32;
    let constant = -2.0 * consts::PI / n;
    for freq in 0..sample_rate {
        let freq = freq as f32;
        let mut res = Complex::new(0.0, 0.0);
        for (n, sample) in signal.iter().enumerate(){
            let theta = constant * freq * n as f32;
            res = res + sample * Complex::new(theta.cos(), theta.sin());
        }

        dft.push(res);
    }

    dft
}
