
use std::fmt;

fn error(expected: &[f32], actual: &[f32]) -> Vec<f32>{
    assert_eq!(expected.len(), actual.len());
    let error = actual.iter().zip(expected.iter()).map(|(a,e)| (a-e).abs()).collect::<Vec<f32>>();
    error
}
fn percent_error(expected: &[f32], actual: &[f32]) -> Vec<f32>{
    assert_eq!(expected.len(), actual.len());
    let error = error(expected, actual);
    let percent_error = error.iter().zip(expected.iter()).map(|(error, expected)| error/expected).collect::<Vec<f32>>();
    percent_error
}

pub fn assert_within_percent_error(expected: &[f32], actual: &[f32], percent_error_allowed: f32){
    let percent_error = percent_error(expected, actual);
    let percent_error_allowed = vec![percent_error_allowed; expected.len()];
    assert!(percent_error <= percent_error_allowed, "percent error is greater than allowed amount");
}

#[test]
fn test_percent_error(){
    let percent_error = percent_error(&[100.0], &[50.0]);
    assert_eq!(percent_error, &[0.5]);
}

#[test]
#[should_panic]
fn test_percent_error_assert_should_panic(){
    assert_within_percent_error(&[100.0], &[50.0], 0.2);
}

#[test]
fn test_percent_error_assert_should_not_panic(){
    assert_within_percent_error(&[100.0], &[20.0], 0.8);
}
