pub trait Organism {
    fn fitness(&self) -> f32;
    fn mutate(&mut self);
    fn cross_over(&self, other: &Self) -> Self
    where
        Self: Sized;
}
