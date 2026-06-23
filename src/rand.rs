unsafe extern "C" {
    fn rand() -> i32;
}

fn rand_uniform() -> f32 {
    unsafe { rand() }.rem_euclid(32768) as f32/32768.
}

pub fn rand_normal() -> f32 {
    ((rand_uniform()*2.-1.).atanh()*1.1).clamp(-3.3, 3.3)
}
