use super::organism::Organism;
use itertools::Itertools;
use rand::{seq::SliceRandom, Rng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TspSolution {
    pub path: Vec<usize>,
}
impl TspSolution {
    pub fn new(nodes: usize) -> Self {
        TspSolution {
            path: (0..nodes).collect(),
        }
    }
}

#[derive(Clone)]
pub struct TspProblem {
    pub graph_weights: Arc<Vec<Vec<f32>>>,
}
impl TspProblem {
    pub fn new(graph_weights: Arc<Vec<Vec<f32>>>) -> Self {
        TspProblem {
            graph_weights: graph_weights,
        }
    }
}

pub struct TSP {
    map: TspProblem,
    solution: TspSolution,
}

impl TSP {
    pub fn new(graph_weights: Arc<Vec<Vec<f32>>>, solution: TspSolution) -> Self {
        let len = graph_weights.len();
        TSP {
            map: TspProblem::new(graph_weights),
            solution,
        }
    }

    pub fn new_with_random_path(graph_weights: Arc<Vec<Vec<f32>>>) -> Self {
        let mut path = (0..graph_weights.len()).collect::<Vec<usize>>();
        path.shuffle(&mut rand::thread_rng());

        TSP {
            map: TspProblem { graph_weights },
            solution: TspSolution { path },
        }
    }

    pub fn get_path(&self) -> &Vec<usize> {
        &self.solution.path
    }

    pub fn get_solution(&self) -> &TspSolution {
        &self.solution
    }

    pub fn get_map(&self) -> &TspProblem {
        &self.map
    }
}

impl Clone for TSP {
    fn clone(&self) -> Self {
        TSP {
            map: self.map.clone(),
            solution: self.solution.clone(),
        }
    }
}

impl Organism for TSP {
    fn fitness(&self) -> f32 {
        let mut cost = self
            .solution
            .path
            .iter()
            .zip(self.solution.path.iter().skip(1))
            .map(|(a, b)| self.map.graph_weights[*a][*b])
            .sum();

        if self.solution.path.iter().unique().count() != self.map.graph_weights.len() {
            cost = f32::INFINITY;
        }

        return cost;
    }

    fn mutate(&mut self) {
        let first_index = rand::thread_rng().gen_range(0..self.solution.path.len());
        let second_index = rand::thread_rng().gen_range(0..self.solution.path.len());

        self.solution.path.swap(first_index, second_index);
    }

    fn cross_over(&self, other: &Self) -> Self
    where
        Self: Sized,
    {
        let mut new_path = vec![0; self.solution.path.len()];
        let mut rng = rand::thread_rng();

        let start_index = rng.gen_range(0..self.solution.path.len());
        let end_index = rng.gen_range(start_index..self.solution.path.len());

        new_path[0..start_index].clone_from_slice(&self.solution.path[0..start_index]);
        new_path[start_index..end_index]
            .clone_from_slice(&other.solution.path[start_index..end_index]);
        new_path[end_index..self.solution.path.len()]
            .clone_from_slice(&self.solution.path[end_index..self.solution.path.len()]);

        TSP {
            map: self.map.clone(),
            solution: TspSolution { path: new_path },
        }
    }
}
