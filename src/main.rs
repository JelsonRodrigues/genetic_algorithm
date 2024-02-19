use bit::BitIndex;
use once_cell::sync::Lazy;
use rand::distributions::{Distribution, Uniform};
use rayon::prelude::*;

const ITERATIONS: usize = 50;
const KEEP_BEST_PERCENT: f64 = 0.4;
const NUMBER_OF_INDIVIDUALS_PER_POPULATION: usize = 10000;
const ELITE: usize = 8;
const MUTATION_RATE: f32 = 0.1;

static NUMBER_OF_BEST_INDIVIDUALS: Lazy<usize> = Lazy::new(|| {
    (NUMBER_OF_INDIVIDUALS_PER_POPULATION as f64 * KEEP_BEST_PERCENT.clamp(0., 1.)) as usize
});

fn main() {
    let mut generation: Vec<LinearSystem> = Vec::new();

    for _ in 0..NUMBER_OF_INDIVIDUALS_PER_POPULATION {
        let x: f32 = rand::random();
        let y: f32 = rand::random();

        generation.push(LinearSystem { x: x, y: y, z: 0.0 });
    }

    (0..ITERATIONS).for_each(|iter| {
        let mut gen = generation
            .par_iter()
            .map(|individual| (individual.fitness(), individual))
            .collect::<Vec<(f32, &LinearSystem)>>();

        gen.par_sort_by(|a, b| a.0.total_cmp(&b.0));
        println!("Iteration: {}", iter);
        println!("Best: {} x:{} y:{}", gen[0].0, gen[0].1.x, gen[0].1.y);

        let mut next_generation = gen[ELITE..*NUMBER_OF_BEST_INDIVIDUALS]
            .par_chunks_exact(2)
            .map(|chunck| {
                let (_, first) = chunck[0];
                let (_, second) = chunck[1];
                let (mut first, mut second) = first.cross_over(second);
                if rand::random::<f32>() < MUTATION_RATE {
                    first.mutate();
                    second.mutate();
                }

                vec![first, second]
            })
            .reduce(
                || Vec::new(),
                |mut acc, x| {
                    acc.extend(x);
                    // println!("{:?}", acc);

                    acc
                },
            );

        (0..NUMBER_OF_INDIVIDUALS_PER_POPULATION - next_generation.len()).for_each(|index| {
            next_generation.push(LinearSystem {
                x: gen[index].1.x,
                y: gen[index].1.y,
                z: gen[index].1.z,
            });
        });

        generation = next_generation;
    });

    for i in 0..5 {
        println!("{i} ({}, {})", generation[0].x, generation[0].y);
    }
}

pub trait Organism {
    fn fitness(&self) -> f32;
    fn mutate(&mut self);
    fn cross_over(&self, other: &Self) -> (Self, Self)
    where
        Self: Sized;
}

#[derive(Debug)]
struct LinearSystem {
    x: f32,
    y: f32,
    z: f32,
}

impl Organism for LinearSystem {
    fn fitness(&self) -> f32 {
        let mut fitness = (self.x.powi(2) * 2.0 + self.y * -3.0 - 25.0).powi(2);
        if !fitness.is_finite() {
            fitness = f32::INFINITY;
        }
        fitness
    }

    fn mutate(&mut self) {
        self.x *= rand::random::<f32>() * 3.0 - 1.5;
        self.y *= rand::random::<f32>() * 3.0 - 1.5;
        // let mut x_bits = self.x.to_bits();

        // let uniform_distribution = rand::distributions::uniform::Uniform::new(0, u32::bit_length());
        // let position = uniform_distribution.sample(&mut rand::thread_rng());
        // x_bits.set_bit(position, !x_bits.bit(position));

        // self.x = unsafe { std::mem::transmute::<u32, f32>(x_bits) };

        // let mut y_bits = self.y.to_bits();

        // let position = uniform_distribution.sample(&mut rand::thread_rng());
        // y_bits.set_bit(position, !y_bits.bit(position));

        // self.y = unsafe { std::mem::transmute::<u32, f32>(y_bits) };
    }

    fn cross_over(&self, other: &Self) -> (LinearSystem, LinearSystem) {
        let uniform_distribution = rand::distributions::uniform::Uniform::new(0, u32::bit_length());
        let cross_over_point = uniform_distribution.sample(&mut rand::thread_rng());

        let mut x_bits = self.x.to_bits();
        let mut y_bits = self.y.to_bits();

        let mut other_x_bits = other.x.to_bits();
        let mut other_y_bits = other.y.to_bits();

        let x_mask = u32::MAX << cross_over_point;
        let y_mask = u32::MAX << cross_over_point;

        let temp = x_bits & x_mask;
        x_bits = (x_bits & !x_mask) | (other_x_bits & x_mask);
        other_x_bits = (other_x_bits & !x_mask) | temp;

        let temp = y_bits & y_mask;
        y_bits = (y_bits & !y_mask) | (other_y_bits & y_mask);
        other_y_bits = (other_y_bits & !y_mask) | temp;

        (
            LinearSystem {
                x: unsafe { std::mem::transmute::<u32, f32>(x_bits) },
                y: unsafe { std::mem::transmute::<u32, f32>(y_bits) },
                z: self.z,
            },
            LinearSystem {
                x: unsafe { std::mem::transmute::<u32, f32>(other_x_bits) },
                y: unsafe { std::mem::transmute::<u32, f32>(other_y_bits) },
                z: other.z,
            },
        )
    }
}
