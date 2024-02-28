use crate::organism::Organism;
use rand::distributions::uniform::UniformSampler;
use rayon::prelude::*;

pub fn ga_iteraration<T>(
    population: Vec<T>,
    mutation_rate: f32,
    crossover_rate: f32,
    elite_size: usize,
) -> Vec<T>
where
    T: Organism + Clone + Sync + Send + Sized,
{
    let distribution = rand::distributions::uniform::UniformFloat::<f32>::new(0.0, 1.0);

    // Evaluate the population
    let mut evaluated_population = population
        .par_iter()
        .map(|individual| (individual.fitness(), individual))
        .collect::<Vec<(f32, &T)>>();

    // Select the best individuals to reproduce
    evaluated_population.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    let mut new_population = evaluated_population[elite_size..]
        .par_windows(2)
        .map(|window| {
            let first = window[0].1.clone();
            let second = window[1].1;

            if distribution.sample(&mut rand::thread_rng()) < crossover_rate {
                let child = first.cross_over(second);
                return child;
            }

            return first.clone();
        })
        .collect::<Vec<T>>();

    // Mutate the new_population
    new_population.par_iter_mut().for_each(|child| {
        if distribution.sample(&mut rand::thread_rng()) < mutation_rate {
            child.mutate();
        }
    });

    // Return the new population, including the elite
    new_population.extend(
        evaluated_population[..=elite_size]
            .iter()
            .cloned()
            .map(|(_, individual)| individual.clone()),
    );

    assert_eq!(new_population.len(), population.len());

    new_population
}
