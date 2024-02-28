use super::organism::Organism;
use itertools::Itertools;
use rand::{seq::SliceRandom, Rng};
use std::sync::Arc;

pub struct TSP {
    graph_weights: Arc<Vec<Vec<f32>>>,
    path: Vec<usize>,
}

impl TSP {
    pub fn new(graph_weights: Vec<Vec<f32>>) -> Self {
        let len = graph_weights.len();
        TSP {
            graph_weights: Arc::new(graph_weights),
            path: (0..len).collect(),
        }
    }

    pub fn new_with_random_path(graph_weights: Arc<Vec<Vec<f32>>>) -> Self {
        let mut path = (0..graph_weights.len()).collect::<Vec<usize>>();
        path.shuffle(&mut rand::thread_rng());

        TSP {
            graph_weights,
            path,
        }
    }

    pub fn get_path(&self) -> &Vec<usize> {
        &self.path
    }
}

impl Clone for TSP {
    fn clone(&self) -> Self {
        TSP {
            graph_weights: self.graph_weights.clone(),
            path: self.path.clone(),
        }
    }
}

impl Organism for TSP {
    fn fitness(&self) -> f32 {
        if self.path.iter().unique().count() != self.path.len() {
            return f32::INFINITY;
        }

        self.path
            .iter()
            .zip(self.path.iter().skip(1))
            .map(|(a, b)| self.graph_weights[*a][*b])
            .sum()
    }

    fn mutate(&mut self) {
        let first_index = rand::thread_rng().gen_range(0..self.path.len());
        let second_index = rand::thread_rng().gen_range(0..self.path.len());

        self.path.swap(first_index, second_index);
    }

    fn cross_over(&self, other: &Self) -> Self
    where
        Self: Sized,
    {
        let mut new_path = vec![0; self.path.len()];
        let mut rng = rand::thread_rng();

        let start_index = rng.gen_range(0..self.path.len());
        let end_index = rng.gen_range(start_index..self.path.len());

        new_path[0..start_index].clone_from_slice(&self.path[0..start_index]);
        new_path[start_index..end_index].clone_from_slice(&other.path[start_index..end_index]);
        new_path[end_index..self.path.len()]
            .clone_from_slice(&self.path[end_index..self.path.len()]);

        TSP {
            graph_weights: self.graph_weights.clone(),
            path: new_path,
        }
    }
}
