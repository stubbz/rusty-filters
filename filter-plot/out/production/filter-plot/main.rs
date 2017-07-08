extern crate gnuplot;
extern crate num;
mod dsp;
mod slice;
mod polynomial;
use polynomial::Polynomial;
use polynomial::ComplexPolynomial;
use dsp::dft;
use dsp::bilinear_transform::BilinearTransform;
use slice::assert_within_percent_error;
use num::complex::Complex;
use num::traits::One;
use num::traits::Zero;

fn main() {
    use std::ops::Range;
    use gnuplot::{Figure, Caption, Color,AxesCommon,AutoOption,Coordinate,ArrowType,ArrowheadType,Axes2D,PointSize,PointSymbol};
    use std::f32::consts;

    plot_freq_response();

    let transform_2 = BilinearTransform::new(2);
    //let lpf2 = butterworth_lpf2(NormalizedFrequency(0.5), QualityFactor(0.5), &transform_2);
    //plot_responses(&lpf2);

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok().expect("pause");

    fn plot_responses(DigitalFilter: &DigitalFilter){
        let n_samples = 20;
        let xs = (0..n_samples).collect::<Vec<_>>();
        let mut fg = Figure::new();

        let step = DigitalFilter.step_response(n_samples);
        let impulse = DigitalFilter.impulse_response(n_samples);

        fg.axes2d()
          .points(xs.iter(), step.iter(), &[Color("blue"), PointSymbol('O'), Caption("step response")])
          .points(xs.iter(), impulse.iter(), &[Color("green"), PointSymbol('O'), Caption("impulse response")]);
        fg.show();
    }


    fn plot_freq_response(){
        let transform_2 = BilinearTransform::new(2);
        let transform_3 = BilinearTransform::new(3);
        let transform_4 = BilinearTransform::new(4);
        let transform_5 = BilinearTransform::new(5);


        let sample_rate = SampleRate(44100);
        let NyquistFrequency(nyquist_frequency) = sample_rate.to_nyquist_frequency();
        let xs = (0..nyquist_frequency).map(|freq| freq as f32).collect::<Vec<_>>();

        let mut fg = Figure::new();

        {
            let axes = &mut fg.axes2d();
            fn add_line(axes: &mut Axes2D, xs: &[f32], cutoff: f32, quality_factor: f32, sample_rate: SampleRate, transform_2: &BilinearTransform, transform_3: &BilinearTransform, transform_4: &BilinearTransform, transform_5: &BilinearTransform){
                let NyquistFrequency(nyquist_frequency) = sample_rate.to_nyquist_frequency();
                //let lpf4 = butterworth_lpf4(NormalizedFrequency(cutoff), QualityFactor(quality_factor), &transform_4).frequency_response(sample_rate);
                //let hpf4 = butterworth_hpf4(NormalizedFrequency(cutoff), QualityFactor(quality_factor)).frequency_response(sample_rate);
                //let lpf2 = butterworth_lpf2(NormalizedFrequency(cutoff), QualityFactor(quality_factor), &transform_2).frequency_response(sample_rate);
                //let hpf2 = butterworth_hpf2(NormalizedFrequency(cutoff), QualityFactor(quality_factor), &transform_2).frequency_response(sample_rate);

                //let c_lpf3 = chebyshev1_lpf(NormalizedFrequency(cutoff), 1.0, &transform_3).frequency_response(sample_rate);
                //let c2_lpf5 = chebyshev2_lpf(NormalizedFrequency(cutoff), 1.0, &transform_5).frequency_response(sample_rate);
                let peak = peak_eq(NormalizedFrequency(cutoff), -6.0, quality_factor, &transform_2).frequency_response(sample_rate);
                let peak2 = peak_eq(NormalizedFrequency(cutoff), 6.0, quality_factor, &transform_2).frequency_response(sample_rate);

                axes
                    //.lines(xs.iter(), c_lpf3.iter().map(|y| y.from_voltage_ratio_to_db()), &[Color("black")])
//                    .lines(xs.iter(), c2_lpf5.iter().map(|y| y.from_voltage_ratio_to_db()), &[Color("black")])
                    .lines(xs.iter(), peak.iter().map(|y| -y.from_voltage_ratio_to_db()), &[Color("black")])
                .lines(xs.iter(), peak2.iter().map(|y| y.from_voltage_ratio_to_db()), &[Color("black")])

                //.lines(xs.iter(), lpf2.iter().map(|y| y.from_voltage_ratio_to_db()), &[Color("blue")])
                    //.lines(xs.iter(), lpf4.iter().map(|y| y.from_voltage_ratio_to_db()), &[Color("black")])
                    //.lines(xs.iter(), hpf2.iter().map(|y| y.from_voltage_ratio_to_db()), &[Color("blue")])
                    //.lines(xs.iter(), hpf4.iter().map(|y| y.from_voltage_ratio_to_db()), &[Color("black")])
                    .arrow(Coordinate::Axis(cutoff as f64 * nyquist_frequency as f64), Coordinate::Axis(9999f64), Coordinate::Axis(cutoff as f64 * nyquist_frequency as f64), Coordinate::Axis(-9999f64), &[ArrowType(ArrowheadType::NoArrow)]);
            }
            //should limit between 0.1 and 18.0
            let qualities = vec![0.25,//-12db
                                 0.5, //-6db
                                 1.0, //0db
                                 2.0, //+6db
                                 4.0 //+12db
                             ];
            let qualities = vec![1.0];
            let cutoffs = vec![0.5, 0.25, 0.75];//vec![0.01, 0.1, 0.5, 1.0];
            for quality in qualities{
                for cutoff in &cutoffs{
                    add_line(axes, &xs, *cutoff, quality, sample_rate, &transform_2, &transform_3, &transform_4, &transform_5);

                }
            }

            axes
            //.set_x_log(Some(2.0))
            .set_y_range(AutoOption::Fix(-48.0), AutoOption::Fix(20.0))
            .set_x_range(AutoOption::Fix(0.01), AutoOption::Fix(nyquist_frequency as f64));
        }

        fg.show();

    }


}


#[derive(Debug, Copy, Clone)]
struct NormalizedFrequency(f32);

impl fmt::Display for NormalizedFrequency {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let &NormalizedFrequency(omega) = self;
        write!(f, "normalized_frequency={}", omega)
    }
}
impl NormalizedFrequency {
    fn prewarp(&self) -> f32{
        use std::f32::consts;

        let &NormalizedFrequency(omega) = self;
        let big_omega = if omega == 0.0 {//prewarping
            0.0
        }else{
            let t = 2.0;
            1.0 / (omega * consts::PI / t).tan()
        };
        big_omega
    }
}



#[derive(Debug, Copy, Clone)]
struct DampingFactor(f32);

#[derive(Debug, Copy, Clone)]
struct QualityFactor(f32);


impl QualityFactor {
    #[inline]
    fn to_damping_factor(&self) -> DampingFactor{
        let &QualityFactor(quality_factor) = self;
        let damping_factor =  1.0 / (2.0 * quality_factor);
        DampingFactor(damping_factor)
    }
}

#[derive(Debug, Copy, Clone)]
struct SampleRate(usize);
#[derive(Debug, Copy, Clone)]
struct NyquistFrequency(usize);

impl SampleRate{
    fn to_nyquist_frequency(&self) -> NyquistFrequency {
        let &SampleRate(sample_rate) = self;
        NyquistFrequency(sample_rate/2)
    }
}


#[derive(Debug)]
struct DigitalFilterPrototype {
    zeros: Vec<Complex<f32>>,
    b_coefficients: Vec<f32>,

    poles: Vec<Complex<f32>>,
    a_coefficients: Vec<f32>,

    gain: f32,

    order: usize,
}

impl DigitalFilterPrototype{
    pub fn new(zeros: Vec<Complex<f32>>, poles: Vec<Complex<f32>>, gain: f32, order: usize) -> Self{



        let b_coefficients = {
            let mut polies = Vec::new();
            for zero in &zeros {
                let mut poly = ComplexPolynomial::new();

                poly.set_coefficient(1, Complex::new(1.0, 0.0));
                poly.set_coefficient(0, -zero);

                polies.push(poly);
            }

            let polies_multiplied = Polynomial::multiply_polynomials(&polies);
            let mut b_coefficients = polies_multiplied.to_vector();
            for missing_zero in 1..(poles.len() - zeros.len() + 1){
                b_coefficients.push(Complex::zero());
            }
            b_coefficients.reverse();
            b_coefficients.iter().map(|a| a.re * gain).collect::<Vec<_>>()
        };


        let a_coefficients = {
            let mut polies = Vec::new();
            for pole in &poles {
                let mut poly = ComplexPolynomial::new();
                poly.set_coefficient(1, Complex::new(1.0, 0.0));
                poly.set_coefficient(0, -pole);
                polies.push(poly);
            }

            let polies_multiplied = Polynomial::multiply_polynomials(&polies);

            let mut a_coefficients = polies_multiplied.to_vector();
            a_coefficients.reverse();
            a_coefficients.iter().map(|a| a.re.abs()).collect::<Vec<_>>()
        };


        DigitalFilterPrototype {
            zeros: zeros,
            b_coefficients: b_coefficients,

            poles: poles,
            a_coefficients: a_coefficients,

            gain: gain,

            order: order,
        }
    }

    pub fn to_digital_filter(&self, normalized_frequency: NormalizedFrequency, transform: &BilinearTransform) -> DigitalFilter{ //todo make this result
        let big_omega = normalized_frequency.prewarp();
        let feed_forward_coefficients = transform.transform(&self.b_coefficients, big_omega);
        let feed_back_coefficients = transform.transform(&self.a_coefficients, big_omega);
        let z_filter = DigitalFilter::new(feed_forward_coefficients, feed_back_coefficients, normalized_frequency);
        println!("{}", z_filter);
        z_filter
    }
}


struct DigitalFilter {
    // sometimes referred to as "b sub k"
    feed_forward_coefficients: Vec<f32>,

    // sometimes referred to as "a sub k"
    feed_back_coefficients: Vec<f32>,

    normalized_frequency: NormalizedFrequency,
}


trait Decibel {
    fn from_db_to_voltage_ratio(&self) -> Self;
    fn from_db_to_power_ratio(&self) -> Self;
}
trait VoltageRatio {
    fn from_voltage_ratio_to_db(&self) -> Self;
}
trait PowerRatio{
    fn from_power_ratio_to_db(&self) -> Self;
}
impl VoltageRatio for f32{
    #[inline]
    fn from_voltage_ratio_to_db(&self) -> Self {
        20.0 * self.abs().log(10.0)
    }
}
impl PowerRatio for f32{
    #[inline]
    fn from_power_ratio_to_db(&self) -> Self{
        10.0 * self.abs().log(10.0)
    }
}
impl Decibel for f32{
    #[inline]
    fn from_db_to_voltage_ratio(&self) -> Self{
        (10.0 as f32).powf(self / 20.0)
    }

    #[inline]
    fn from_db_to_power_ratio(&self) -> Self{
        (10.0 as f32).powf(self / 10.0)
    }
}

use std::fmt;



#[test]
fn test_butterworth_lpf2_cutoff(){
    fn test(normalized_frequency: NormalizedFrequency, feed_back_coefficients: Vec<f32>){
        let transform_2 = BilinearTransform::new(2);
        let DigitalFilter = butterworth_lpf2(normalized_frequency, QualityFactor(0.5), &transform_2);
        assert_within_percent_error(&DigitalFilter.feed_back_coefficients, &feed_back_coefficients, 0.001);
        assert_within_percent_error(&DigitalFilter.feed_forward_coefficients, &[1.0, 2.0, 1.0], 0.001);
    }
    test(NormalizedFrequency(0.75), vec![1.7573593, 1.6568543, 0.58578646]);
    test(NormalizedFrequency(0.25), vec![10.24264, -9.656853, 3.414213]);
    test(NormalizedFrequency(0.5), vec![3.4142137, 0.0, 0.58578646]);
    test(NormalizedFrequency(1.0), vec![1.0, 2.0, 1.0]);
    test(NormalizedFrequency(0.0), vec![1.0, -2.0, 1.0]);
}
#[test]
fn test_butterworth_lpf4_cutoff(){
    fn test(normalized_frequency: NormalizedFrequency, feed_back_coefficients: Vec<f32>){
        let transform_4 = BilinearTransform::new(4);
        let DigitalFilter = butterworth_lpf4(normalized_frequency, QualityFactor(0.5), &transform_4);
        assert_within_percent_error(&DigitalFilter.feed_back_coefficients, &feed_back_coefficients, 0.001);
        assert_within_percent_error(&DigitalFilter.feed_forward_coefficients, &vec![1.0, 4.0, 6.0, 4.0, 1.0], 0.001);
    }
    test(NormalizedFrequency(0.75), vec![2.883325, 5.6756177, 5.0050507, 2.0888846, 0.3471222]);
    test(NormalizedFrequency(0.25), vec![97.94816, -192.80386, 170.02435, -70.96056, 11.791934]);
    test(NormalizedFrequency(0.5), vec![10.640467, 0.0, 5.1715717, 0.0, 0.18696158]);
    test(NormalizedFrequency(1.0), vec![1.0, 4.0, 6.0, 4.0, 1.0]);
    test(NormalizedFrequency(0.0), vec![1.0, -4.0, 6.0, -4.0, 1.0]);
}

use std::f32::consts;

fn quadratic_eq(a: f32, b: f32, c: f32) -> Vec<Complex<f32>>{
    let a = Complex::new(a, 0.0);
    let b = Complex::new(b, 0.0);
    let c = Complex::new(c, 0.0);

    let denom = Complex::new(2.0, 0.0) * a;
    let term_1 = - b / denom;
    let term_2 = ((b*b) - Complex::new(4.0, 0.0) * a * c).sqrt() / denom;

    let mut res = Vec::new();
    res.push(term_1 + term_2);
    res.push(term_1 - term_2);
    res
}

fn peak_eq_prototype(db: f32, q: f32) -> DigitalFilterPrototype {
    let g = db.from_db_to_power_ratio();
    println!("{}", g);
    let b = 1.0 / (q) * 2.0;

    //modeled from https://ccrma.stanford.edu/~jos/fp/Peaking_Equalizers.html
    //had to factor out2 poles and zeros
    let zeros = quadratic_eq(1.0, g * b, 1.0);
    let poles = quadratic_eq(1.0, b, 1.0);

    let gain = 1.0;
    let order = 2;
    let s_filter = DigitalFilterPrototype::new(zeros, poles, gain, order);
    s_filter
}

fn peak_eq(normalized_frequency: NormalizedFrequency, db: f32, q: f32, transform: &BilinearTransform) -> DigitalFilter {
    let s_filter = peak_eq_prototype(db, q);
    println!("{:?}", s_filter.b_coefficients);
    println!("{:?}", s_filter.a_coefficients);
    let z_filter = s_filter.to_digital_filter(normalized_frequency, transform);
    z_filter
}


fn chebyshev1_pole(m: f32, n:f32, epsillon: f32) -> Complex<f32>{
    let theta = consts::FRAC_PI_2 * (2.0 * m - 1.0) / n;
    let re = -1.0 * ((1.0 / epsillon).asinh() / n).sinh() * theta.sin();
    let im = ((1.0 / epsillon).asinh() / n).cosh() * theta.cos();
    let pole = Complex::new(re, im);

    pole
}
fn chebyshev1_zero(l: f32, n: f32) -> Complex<f32> {
    let theta = consts::FRAC_PI_2 * (2.0 * l - 1.0) / n;
    let zero = Complex::new(0.0, -theta.cos());
    zero
}
fn chebyshev2_pole(m: f32, n:f32, epsillon: f32) -> Complex<f32> {
    let pole = Complex::one() / chebyshev1_pole(m,n,epsillon);

    pole
}

fn chebyshev2_zero(l:f32, n: f32) -> Complex<f32> {
    let zero = Complex::one() / chebyshev1_zero(l,n);
    println!("{}", zero.is_finite());
    zero
}

fn chebyshev2_lpf_protoype(order: usize, pass_band_ripple_db: f32) -> DigitalFilterPrototype {
    let ripple_factor = 1.0 / (pass_band_ripple_db.from_db_to_power_ratio() - 1.0).sqrt();
    println!("ripple = {}", ripple_factor);
    let num_poles = order as f32;


    let mut zeros = Vec::new();
    for zero_number in 1..(order + 1) {
        let zero = chebyshev2_zero(zero_number as f32, num_poles);
        println!("zero {} = {}", zero_number, zero);

        //filter out zeros at infinity, these do not affect the response
        //todo remove this hack
        if(zero.is_finite() && zero.im < 9999.0 && zero.im > -9999.0){//the zero isn't truly infinity because it's being calced as like 1/.000001 instead of 1/0
            zeros.push(zero);
        }
    }


    let mut poles = Vec::new();
    for pole_number in 1..(order + 1) {//todo calculate left plane poles correctly
        let pole = chebyshev2_pole(pole_number as f32, num_poles, ripple_factor);
        println!("pole {} = {}", pole_number, pole);
        poles.push(pole);
    }

    //optimal gain
    //http://ocw.mit.edu/courses/mechanical-engineering/2-161-signal-processing-continuous-and-discrete-fall-2008/lecture-notes/lecture_07.pdf
    //search for unity gain
    let gain = {//todo perfect this
        let gain = - poles.iter().fold(Complex::one(), |a,b| a*b) / zeros.iter().fold(Complex::one(), |a,b| a*b);
        gain.re
    };
    println!("gain = {}", gain);

    let s_filter = DigitalFilterPrototype::new(zeros, poles, gain, order);
    s_filter
}

fn chebyshev2_lpf(normalized_frequency: NormalizedFrequency, pass_band_ripple_db: f32, transform: &BilinearTransform) -> DigitalFilter {//todo return result
    let order = transform.order();
    let s_filter = chebyshev2_lpf_protoype(order, pass_band_ripple_db);
    let z_filter = s_filter.to_digital_filter(normalized_frequency, transform);
    z_filter
}

fn chebyshev1_lpf(normalized_frequency: NormalizedFrequency, pass_band_ripple_db: f32, transform: &BilinearTransform) -> DigitalFilter {//todo return result
    use std::f32::consts;

    let ripple_factor =  (pass_band_ripple_db.from_power_ratio_to_db() - 1.0).sqrt();
    println!("ripple = {}", ripple_factor);
    let order = transform.order();

    //optimal gain
    //http://www.ece.uah.edu/courses/ee426/Chebyshev.pdf
    let gain = {
        if order % 2 == 0{
            1.0 / (1.0 + ripple_factor.powi(2)).sqrt()
        }else{
            1.0
        }
    };



    let num_poles = order as f32;
    let mut poles = Vec::new();
    for pole_number in 1..(order + 1) {
        let pole = chebyshev1_pole(pole_number as f32, num_poles, ripple_factor);
        println!("pole {} = {}", pole_number, pole);
        poles.push(pole);
    }

    let a_vec = {
        let mut polies = Vec::new();
        for pole in &poles {
            let mut poly = ComplexPolynomial::new();
            poly.set_coefficient(1, Complex::new(1.0, 0.0));
            poly.set_coefficient(0, -pole);
            polies.push(poly);
        }

        let polies_multiplied = Polynomial::multiply_polynomials(&polies);

        let mut denom = polies_multiplied.to_vector();
        denom.reverse();
        denom.iter().map(|a| a.re).collect::<Vec<f32>>()
    };

    let mut b_vec = vec![0.0; a_vec.len()];
    b_vec[poles.len()] = gain * poles.iter().fold(Complex::one(), |a,b| -a*b).re;
    println!("{:?}", a_vec);
    println!("{:?}", b_vec);

    let big_omega = normalized_frequency.prewarp();
    let feed_back_coefficients = transform.transform(&a_vec, big_omega);
    let feed_forward_coefficients = transform.transform(&b_vec, big_omega);
    let z_filter = DigitalFilter::new(feed_forward_coefficients, feed_back_coefficients, normalized_frequency);
    println!("{}", z_filter);
    z_filter
}

#[inline]
fn butterworth_hpf2(normalized_frequency: NormalizedFrequency, quality_factor: QualityFactor, transform: &BilinearTransform) -> DigitalFilter{//todo return result
    use std::f32::consts;

    let big_omega = normalized_frequency.prewarp();

    let feed_back_coefficients = {
        let DampingFactor(damping_factor) =  quality_factor.to_damping_factor();
        let middle = consts::SQRT_2 * damping_factor;
        transform.transform(&[1.0, middle, 1.0], big_omega)
    };

    let feed_forward_coefficients = transform.transform(&[1.0, 0.0, 0.0], big_omega);
    let z_filter = DigitalFilter::new(feed_forward_coefficients,feed_back_coefficients, normalized_frequency);
    println!("{}", z_filter);
    z_filter
}

#[inline]
fn butterworth_lpf(normalized_frequency: NormalizedFrequency, transform: &BilinearTransform) -> DigitalFilter {
    let big_omega = normalized_frequency.prewarp();

    let feed_back_coefficients = {
        let poly = normalized_butterworth_polynomial(transform.order() as u32);
        let mut fb = poly.to_vector();
        fb.reverse();
        fb
    };

    let mut a_vector = vec![0.0; transform.order() + 1];
    a_vector[0] = 1.0;
    let feed_forward_coefficients = transform.transform(&a_vector, big_omega);
    let z_filter = DigitalFilter::new(feed_forward_coefficients, feed_back_coefficients, normalized_frequency);
    println!("{}", z_filter);
    z_filter
}



#[inline]
fn normalized_butterworth_polynomial(order: u32) -> Polynomial<f32> {
    fn butter_poly(k : f32, n: f32) -> Polynomial<f32>{
        use std::f32::consts;

        let mut poly = Polynomial::new();
        poly.set_coefficient(2, 1.0);
        poly.set_coefficient(1, -2.0 * (2.0 * k + n - 1.0 * consts::PI / (2.0 * n)));
        poly.set_coefficient(0, 1.0);
        poly
    }

    let mut polynomials = Vec::new();
    if order % 2 == 0 {
        for k in 1..(order/2){
            polynomials.push(butter_poly(k as f32, order as f32));
        }
    }else{
        let mut s_plus_1 = Polynomial::new();
        s_plus_1.set_coefficient(1, 1.0);
        s_plus_1.set_coefficient(0, 1.0);
        polynomials.push(s_plus_1);

        for k in 1..((order-1)/2){
            polynomials.push(butter_poly(k as f32, order as f32));
        }
    }

    Polynomial::multiply_polynomials(&polynomials)
}

#[inline]
fn butterworth_lpf2(normalized_frequency: NormalizedFrequency, quality_factor: QualityFactor, transform: &BilinearTransform) -> DigitalFilter{//todo return result
    use std::f32::consts;

    let big_omega = normalized_frequency.prewarp();

    let feed_back_coefficients =  {
        let DampingFactor(damping_factor) =  quality_factor.to_damping_factor();
        let middle = consts::SQRT_2 * damping_factor;
        transform.transform(&[1.0, middle, 1.0], big_omega)
    };

    let feed_forward_coefficients = transform.transform(&[0.0, 0.0, 1.0], big_omega);
    let z_filter = DigitalFilter::new(feed_forward_coefficients,feed_back_coefficients, normalized_frequency);
    println!("{}", z_filter);
    z_filter
}


#[inline]
fn butterworth_hpf4(normalized_frequency: NormalizedFrequency, quality_factor: QualityFactor, transform: &BilinearTransform) -> DigitalFilter{//todo return result
    use std::f32::consts;

    let big_omega = normalized_frequency.prewarp();

    let feed_back_coefficients = {
        let DampingFactor(damping_factor) = quality_factor.to_damping_factor();
        let cos_frac_7pi_8 = (7.0 * consts::FRAC_PI_8).cos() * damping_factor.sqrt();
        let cos_frac_5pi_8 = (5.0 * consts::FRAC_PI_8).cos() * damping_factor.sqrt();

        let a0 = 1.0;
        let a1 = (- 2.0 * cos_frac_7pi_8) - (2.0 * cos_frac_5pi_8);
        let a2 = (2.0) + (4.0 * cos_frac_7pi_8 * cos_frac_5pi_8);
        let a3 = a1;
        let a4 = a0;
        transform.transform(&[a0, a1, a2, a3, a4], big_omega)
    };

    let feed_forward_coefficients = transform.transform(&[1.0, 0.0, 0.0, 0.0, 0.0], big_omega);
    let z_filter = DigitalFilter::new(feed_forward_coefficients, feed_back_coefficients, normalized_frequency);
    println!("{}", z_filter);
    z_filter
}

#[inline]
fn butterworth_lpf4(normalized_frequency: NormalizedFrequency, quality_factor: QualityFactor, transform: &BilinearTransform) -> DigitalFilter{//todo return result
    use std::f32::consts;

    let big_omega = normalized_frequency.prewarp();

    let feed_back_coefficients = {
        let DampingFactor(damping_factor) = quality_factor.to_damping_factor();
        let cos_frac_7pi_8 = (7.0 * consts::FRAC_PI_8).cos() * damping_factor.sqrt();
        let cos_frac_5pi_8 = (5.0 * consts::FRAC_PI_8).cos() * damping_factor.sqrt();

        let a0 = 1.0;
        let a1 = (- 2.0 * cos_frac_7pi_8) - (2.0 * cos_frac_5pi_8);
        let a2 = (2.0) + (4.0 * cos_frac_7pi_8 * cos_frac_5pi_8);
        let a3 = a1;
        let a4 = a0;
        transform.transform(&[a0, a1, a2, a3, a4], big_omega)
    };

    let feed_forward_coefficients = transform.transform(&[0.0, 0.0, 0.0, 0.0, 1.0], big_omega);

    let z_filter = DigitalFilter::new(feed_forward_coefficients, feed_back_coefficients, normalized_frequency);
    println!("{}", z_filter);
    z_filter
}

impl DigitalFilter {
    #[inline]
    fn new(feed_forward_coefficients: Vec<f32>, feed_back_coefficients: Vec<f32>, normalized_frequency: NormalizedFrequency) -> Self {
        DigitalFilter {
            feed_forward_coefficients: feed_forward_coefficients,
            feed_back_coefficients: feed_back_coefficients,
            normalized_frequency: normalized_frequency,
        }
    }

    #[inline]
    fn frequency_response(&self, sample_rate: SampleRate) -> Vec<f32> {
        use std::f32;
        use dsp::HasMagnitude;

        let SampleRate(sample_rate) = sample_rate;
        let top =  dft(&self.feed_forward_coefficients, sample_rate);
        let bot = dft(&self.feed_back_coefficients, sample_rate);

        top.iter().zip(bot.iter()).map(|(t,b)| (t/b).magnitude()).collect::<Vec<f32>>()
    }

    #[inline]
    fn process_signal(&self, input: &[f32], output: &mut [f32]){
        //todo return next state?

        #[inline]
        fn get_sample(sample_index: i32, coeff_index: i32, signal: &[f32]) -> f32{
            let index = sample_index - coeff_index;
            if index >= 0 {
                signal[index as usize]
            }else{
                0.0
            }
        }

        println!("{}", self);
        for sample_index in 0..input.len() {
            let mut output_sample = 0.0;

            for (coeff_index, b) in self.feed_forward_coefficients.iter().enumerate(){
                let x = get_sample(sample_index as i32, coeff_index as i32, input);
                output_sample = output_sample + (x * b);
            }

            for (coeff_index, a) in self.feed_back_coefficients.iter().enumerate(){
                let y = get_sample(sample_index as i32, coeff_index as i32, output);
                output_sample = output_sample - (y * a);
            }

            output_sample = output_sample / self.feed_back_coefficients[0];//todo this division could be removed if the DigitalFilter was normalized
            output[sample_index] = output_sample;
        }
    }

    fn impulse_response(&self, length: usize) -> Vec<f32>{
        let mut input = vec![0.0; length];
        input[0] = 1.0;
        let mut output = vec![0.0; length];
        self.process_signal(&input, &mut output);
        output
    }

    fn step_response(&self, length: usize) -> Vec<f32>{
        let input = vec![1.0; length];
        let mut output = vec![0.0; length];
        self.process_signal(&input, &mut output);
        output
    }
}


impl fmt::Display for DigitalFilter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}", self.normalized_frequency);

        for (i,b) in self.feed_forward_coefficients.iter().enumerate() {
            writeln!(f, "b[{}] = {}", i, b);
        }

        for (i,a) in self.feed_back_coefficients.iter().enumerate() {
            writeln!(f, "a[{}] = {}", i, a);
        }

        Ok(())
    }

}
