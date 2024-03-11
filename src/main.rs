pub mod genetic_algorithm;
pub mod organism;
pub mod tsp;

use itertools::Itertools;
use mpi::traits::{Communicator, CommunicatorCollectives, Destination, Root, Source};
use once_cell::sync::Lazy;
use rand::distributions::{uniform::UniformSampler, Distribution, Uniform};
use rayon::{prelude::*, vec};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tsp::{TspSolution, TSP};

use crate::organism::Organism;

const ITERATIONS: usize = 50;
const NUMBER_OF_INDIVIDUALS_PER_POPULATION: usize = 10000;
const ELITE: usize = 20;
const MUTATION_RATE: f32 = 0.1;
const CROSSOVER_RATE: f32 = 0.9;

#[derive(Clone, Serialize, Deserialize)]
enum Message {
    Terminate,
    Population(Vec<TspSolution>),
    MapCreation(Vec<Vec<f32>>),
    EvaluatedPopulation(Vec<(f32, TspSolution)>),
}

const ROOT_PROCESS: i32 = 0;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    println!("Rank: {}, Size: {}", rank, size);
    let portions = NUMBER_OF_INDIVIDUALS_PER_POPULATION / ((size - 1) as usize);

    if rank == ROOT_PROCESS {
        // Initialize and broadcast the map
        let mut tsp = initialize();
        let distribution = rand::distributions::uniform::UniformFloat::<f32>::new(0.0, 1.0);

        println!("Root process is now going to broadcast the map");
        let map = tsp.first().unwrap().get_map().graph_weights.to_vec();
        let graph_weights = Arc::new(map);
        let mut serialized =
            bincode::serialize(&Message::MapCreation(graph_weights.to_vec())).unwrap();

        world
            .process_at_rank(ROOT_PROCESS)
            .broadcast_into(&mut serialized.len());

        world
            .process_at_rank(ROOT_PROCESS)
            .broadcast_into(&mut serialized);

        println!("Map is Broadcasted");

        for i in 0..ITERATIONS {
            println!("Iteration {} BEGIN", i);

            // Scatter the population to the other processes
            println!("Sending pieces of vector to processes");
            // tsp.par_iter()
            //     .map(|value| value.get_solution().clone())
            //     .chunks(portions)
            //     .enumerate()
            tsp.iter()
                .map(|value| value.get_solution().clone())
                .chunks(portions)
                .into_iter()
                .enumerate()
                .for_each(|(i, chunk)| {
                    println!("Now sending to process {}", i + 1,);
                    // let buffer = bincode::serialize(&Message::Population(chunk.to_vec())).unwrap();
                    let buffer =
                        bincode::serialize(&Message::Population(chunk.collect_vec())).unwrap();
                    world.process_at_rank(i as i32 + 1).send(&buffer[..]);
                    println!("sended to process {}", i + 1);
                });
            println!("Send finished");

            println!("Receiving Results from processes");
            // Gather the new population from the other processes
            let mut eval_pop = (1..size)
                .map(|i| {
                    let (buffer, stats) = world.process_at_rank(i).receive_vec();
                    let message = bincode::deserialize::<Message>(&buffer);

                    if let Ok(Message::EvaluatedPopulation(evaluated_population)) = message {
                        evaluated_population
                    } else {
                        panic!("Error receiving evaluated population")
                    }
                })
                .reduce(|mut acc, mut evaluated_population| {
                    acc.append(&mut evaluated_population);
                    acc
                })
                .unwrap();
            println!("Receiving results finished");
            // Sort all the populations

            eval_pop.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

            // Print the best ones

            println!(
                "Iteration {}, Best ones: {:?}",
                i,
                eval_pop[0..10]
                    .iter()
                    .map(|(fit, _)| fit)
                    .collect::<Vec<_>>()
            );

            // Select the best individuals to reproduce
            let mut tsp_population = eval_pop
                .par_iter()
                .cloned()
                .map(|val| (val.0, TSP::new(graph_weights.clone(), val.1)))
                .collect::<Vec<(f32, TSP)>>();

            let mut new_population = tsp_population[ELITE..]
                .par_windows(2)
                .map(|window| {
                    let first = window[0].1.clone();
                    let second = &window[1].1;

                    if distribution.sample(&mut rand::thread_rng()) < CROSSOVER_RATE {
                        let child = first.cross_over(&second);
                        return child;
                    }

                    return first.clone();
                })
                .collect::<Vec<TSP>>();

            // Mutate the new_population
            new_population.par_iter_mut().for_each(|child| {
                if distribution.sample(&mut rand::thread_rng()) < MUTATION_RATE {
                    child.mutate();
                }
            });

            // Return the new population, including the elite
            new_population.extend(
                tsp_population[..=ELITE]
                    .iter()
                    .cloned()
                    .map(|(_, individual)| individual.clone()),
            );

            tsp = new_population;
            println!("TSP size {}", tsp.len());
            println!("Iteration {} END", i);
        }

        tsp.iter()
            .map(|value| value.get_solution().clone())
            .chunks(portions)
            .into_iter()
            .enumerate()
            .for_each(|(i, chunk)| {
                // let buffer = bincode::serialize(&Message::Population(chunk.to_vec())).unwrap();
                let buffer = bincode::serialize(&Message::Population(chunk.collect_vec())).unwrap();
                world.process_at_rank(i as i32 + 1).send(&buffer[..]);
            });

        // Gather the new population from the other processes
        let mut eval_pop = (1..size)
            .map(|i| {
                let (buffer, stats) = world.process_at_rank(i).receive_vec();
                let message = bincode::deserialize::<Message>(&buffer);

                if let Ok(Message::EvaluatedPopulation(evaluated_population)) = message {
                    evaluated_population
                } else {
                    panic!("Error receiving evaluated population")
                }
            })
            .reduce(|mut acc, mut evaluated_population| {
                acc.append(&mut evaluated_population);
                acc
            })
            .unwrap();

        // Sort all the populations

        eval_pop.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        // Print the best ones

        eval_pop[0..10]
            .iter()
            .for_each(|(fit, solution)| println!("Best ones: {:?} -> {:?}", fit, solution));

        (1..size).for_each(|i| {
            let buffer = bincode::serialize(&Message::Terminate).unwrap();
            world.process_at_rank(i).send(&buffer[..]);
        });
    } else {
        let mut bytes = 0;
        world
            .process_at_rank(ROOT_PROCESS)
            .broadcast_into(&mut bytes);

        let mut buffer: Vec<u8> = vec![0; bytes];

        world
            .process_at_rank(ROOT_PROCESS)
            .broadcast_into(&mut buffer);

        let message = bincode::deserialize::<Message>(&buffer);
        println!("Process {} received the message", rank);

        if let Ok(Message::MapCreation(map)) = message {
            let map = Arc::new(map);
            println!("Process {} received the map", rank);
            loop {
                // Receive the population from the root process or a termination signal
                let (buffer, stats) = world.process_at_rank(ROOT_PROCESS).receive_vec();
                let message = bincode::deserialize::<Message>(&buffer);

                println!("Process {} received message, stats {:?}", rank, stats);

                if let Ok(Message::Terminate) = message {
                    println!("Process {} received termination signal", rank);
                    break;
                }

                if let Ok(Message::Population(population)) = message {
                    // Evaluate the fitness function of the population
                    let pop_tsp = population
                        .par_iter()
                        .cloned()
                        .map(|individual| TSP::new(map.clone(), individual))
                        .collect::<Vec<TSP>>();

                    // Return a vec of tuples with the fitness and the individual
                    let evaluated_population = genetic_algorithm::ga_evaluate_population(&pop_tsp)
                        .par_iter()
                        .map(|(fitnes, tsp)| (*fitnes, tsp.get_solution().clone()))
                        .collect::<Vec<(f32, TspSolution)>>();

                    // Send the evaluated population to the root process
                    let serialized =
                        bincode::serialize(&Message::EvaluatedPopulation(evaluated_population))
                            .expect("Failed to serialize the evaluated population");

                    println!("Process {} will send the evaluated population", rank);
                    world.process_at_rank(ROOT_PROCESS).send(&serialized);
                    println!("Process {} sent the evaluated population", rank);
                }
            }
        }
        println!("Process {} is done", rank);
    }
}

fn initialize() -> Vec<TSP> {
    let graph_weights = vec![
        vec![
            0.0, 74.0, 4110.0, 3048.0, 2267.0, 974.0, 4190.0, 3302.0, 4758.0, 3044.0, 3095.0,
            3986.0, 5093.0, 6407.0, 5904.0, 8436.0, 6963.0, 6694.0, 6576.0, 8009.0, 7399.0, 7267.0,
            7425.0, 9639.0, 9230.0, 8320.0, 9300.0, 8103.0, 7799.0,
        ],
        vec![
            74.0, 0.0, 4070.0, 3000.0, 2214.0, 901.0, 4138.0, 3240.0, 4702.0, 2971.0, 3021.0,
            3915.0, 5025.0, 6338.0, 5830.0, 8369.0, 6891.0, 6620.0, 6502.0, 7939.0, 7326.0, 7193.0,
            7351.0, 9571.0, 9160.0, 8249.0, 9231.0, 8030.0, 7725.0,
        ],
        vec![
            4110.0, 4070.0, 0.0, 1173.0, 1973.0, 3496.0, 892.0, 1816.0, 1417.0, 3674.0, 3778.0,
            2997.0, 2877.0, 3905.0, 5057.0, 5442.0, 4991.0, 5151.0, 5316.0, 5596.0, 5728.0, 5811.0,
            5857.0, 6675.0, 6466.0, 6061.0, 6523.0, 6165.0, 6164.0,
        ],
        vec![
            3048.0, 3000.0, 1173.0, 0.0, 817.0, 2350.0, 1172.0, 996.0, 1797.0, 2649.0, 2756.0,
            2317.0, 2721.0, 3974.0, 4548.0, 5802.0, 4884.0, 4887.0, 4960.0, 5696.0, 5537.0, 5546.0,
            5634.0, 7045.0, 6741.0, 6111.0, 6805.0, 6091.0, 5977.0,
        ],
        vec![
            2267.0, 2214.0, 1973.0, 817.0, 0.0, 1533.0, 1924.0, 1189.0, 2498.0, 2209.0, 2312.0,
            2325.0, 3089.0, 4401.0, 4558.0, 6342.0, 5175.0, 5072.0, 5075.0, 6094.0, 5755.0, 5712.0,
            5828.0, 7573.0, 7222.0, 6471.0, 7289.0, 6374.0, 6187.0,
        ],
        vec![
            974.0, 901.0, 3496.0, 2350.0, 1533.0, 0.0, 3417.0, 2411.0, 3936.0, 2114.0, 2175.0,
            3014.0, 4142.0, 5450.0, 4956.0, 7491.0, 5990.0, 5725.0, 5615.0, 7040.0, 6430.0, 6304.0,
            6459.0, 8685.0, 8268.0, 7348.0, 8338.0, 7131.0, 6832.0,
        ],
        vec![
            4190.0, 4138.0, 892.0, 1172.0, 1924.0, 3417.0, 0.0, 1233.0, 652.0, 3086.0, 3185.0,
            2203.0, 1987.0, 3064.0, 4180.0, 4734.0, 4117.0, 4261.0, 4425.0, 4776.0, 4844.0, 4922.0,
            4971.0, 5977.0, 5719.0, 5228.0, 5780.0, 5302.0, 5281.0,
        ],
        vec![
            3302.0, 3240.0, 1816.0, 996.0, 1189.0, 2411.0, 1233.0, 0.0, 1587.0, 1877.0, 1979.0,
            1321.0, 1900.0, 3214.0, 3556.0, 5175.0, 4006.0, 3947.0, 3992.0, 4906.0, 4615.0, 4599.0,
            4700.0, 6400.0, 6037.0, 5288.0, 6105.0, 5209.0, 5052.0,
        ],
        vec![
            4758.0, 4702.0, 1417.0, 1797.0, 2498.0, 3936.0, 652.0, 1587.0, 0.0, 3286.0, 3374.0,
            2178.0, 1576.0, 2491.0, 3884.0, 4088.0, 3601.0, 3818.0, 4029.0, 4180.0, 4356.0, 4469.0,
            4497.0, 5331.0, 5084.0, 4645.0, 5143.0, 4761.0, 4787.0,
        ],
        vec![
            3044.0, 2971.0, 3674.0, 2649.0, 2209.0, 2114.0, 3086.0, 1877.0, 3286.0, 0.0, 107.0,
            1360.0, 2675.0, 3822.0, 2865.0, 5890.0, 4090.0, 3723.0, 3560.0, 5217.0, 4422.0, 4257.0,
            4428.0, 7000.0, 6514.0, 5455.0, 6587.0, 5157.0, 4802.0,
        ],
        vec![
            3095.0, 3021.0, 3778.0, 2756.0, 2312.0, 2175.0, 3185.0, 1979.0, 3374.0, 107.0, 0.0,
            1413.0, 2725.0, 3852.0, 2826.0, 5916.0, 4088.0, 3705.0, 3531.0, 5222.0, 4402.0, 4229.0,
            4403.0, 7017.0, 6525.0, 5451.0, 6598.0, 5142.0, 4776.0,
        ],
        vec![
            3986.0, 3915.0, 2997.0, 2317.0, 2325.0, 3014.0, 2203.0, 1321.0, 2178.0, 1360.0, 1413.0,
            0.0, 1315.0, 2511.0, 2251.0, 4584.0, 2981.0, 2778.0, 2753.0, 4031.0, 3475.0, 3402.0,
            3531.0, 5734.0, 5283.0, 4335.0, 5355.0, 4143.0, 3897.0,
        ],
        vec![
            5093.0, 5025.0, 2877.0, 2721.0, 3089.0, 4142.0, 1987.0, 1900.0, 1576.0, 2675.0, 2725.0,
            1315.0, 0.0, 1323.0, 2331.0, 3350.0, 2172.0, 2275.0, 2458.0, 3007.0, 2867.0, 2935.0,
            2988.0, 4547.0, 4153.0, 3400.0, 4222.0, 3376.0, 3307.0,
        ],
        vec![
            6407.0, 6338.0, 3905.0, 3974.0, 4401.0, 5450.0, 3064.0, 3214.0, 2491.0, 3822.0, 3852.0,
            2511.0, 1323.0, 0.0, 2350.0, 2074.0, 1203.0, 1671.0, 2041.0, 1725.0, 1999.0, 2213.0,
            2173.0, 3238.0, 2831.0, 2164.0, 2901.0, 2285.0, 2397.0,
        ],
        vec![
            5904.0, 5830.0, 5057.0, 4548.0, 4558.0, 4956.0, 4180.0, 3556.0, 3884.0, 2865.0, 2826.0,
            2251.0, 2331.0, 2350.0, 0.0, 3951.0, 1740.0, 1108.0, 772.0, 2880.0, 1702.0, 1450.0,
            1650.0, 4779.0, 4197.0, 2931.0, 4270.0, 2470.0, 2010.0,
        ],
        vec![
            8436.0, 8369.0, 5442.0, 5802.0, 6342.0, 7491.0, 4734.0, 5175.0, 4088.0, 5890.0, 5916.0,
            4584.0, 3350.0, 2074.0, 3951.0, 0.0, 2222.0, 2898.0, 3325.0, 1276.0, 2652.0, 3019.0,
            2838.0, 1244.0, 1089.0, 1643.0, 1130.0, 2252.0, 2774.0,
        ],
        vec![
            6963.0, 6891.0, 4991.0, 4884.0, 5175.0, 5990.0, 4117.0, 4006.0, 3601.0, 4090.0, 4088.0,
            2981.0, 2172.0, 1203.0, 1740.0, 2222.0, 0.0, 684.0, 1116.0, 1173.0, 796.0, 1041.0,
            974.0, 3064.0, 2505.0, 1368.0, 2578.0, 1208.0, 1201.0,
        ],
        vec![
            6694.0, 6620.0, 5151.0, 4887.0, 5072.0, 5725.0, 4261.0, 3947.0, 3818.0, 3723.0, 3705.0,
            2778.0, 2275.0, 1671.0, 1108.0, 2898.0, 684.0, 0.0, 432.0, 1776.0, 706.0, 664.0, 756.0,
            3674.0, 3090.0, 1834.0, 3162.0, 1439.0, 1120.0,
        ],
        vec![
            6576.0, 6502.0, 5316.0, 4960.0, 5075.0, 5615.0, 4425.0, 3992.0, 4029.0, 3560.0, 3531.0,
            2753.0, 2458.0, 2041.0, 772.0, 3325.0, 1116.0, 432.0, 0.0, 2174.0, 930.0, 699.0, 885.0,
            4064.0, 3469.0, 2177.0, 3540.0, 1699.0, 1253.0,
        ],
        vec![
            8009.0, 7939.0, 5596.0, 5696.0, 6094.0, 7040.0, 4776.0, 4906.0, 4180.0, 5217.0, 5222.0,
            4031.0, 3007.0, 1725.0, 2880.0, 1276.0, 1173.0, 1776.0, 2174.0, 0.0, 1400.0, 1770.0,
            1577.0, 1900.0, 1332.0, 510.0, 1406.0, 1002.0, 1499.0,
        ],
        vec![
            7399.0, 7326.0, 5728.0, 5537.0, 5755.0, 6430.0, 4844.0, 4615.0, 4356.0, 4422.0, 4402.0,
            3475.0, 2867.0, 1999.0, 1702.0, 2652.0, 796.0, 706.0, 930.0, 1400.0, 0.0, 371.0, 199.0,
            3222.0, 2611.0, 1285.0, 2679.0, 769.0, 440.0,
        ],
        vec![
            7267.0, 7193.0, 5811.0, 5546.0, 5712.0, 6304.0, 4922.0, 4599.0, 4469.0, 4257.0, 4229.0,
            3402.0, 2935.0, 2213.0, 1450.0, 3019.0, 1041.0, 664.0, 699.0, 1770.0, 371.0, 0.0,
            220.0, 3583.0, 2970.0, 1638.0, 3037.0, 1071.0, 560.0,
        ],
        vec![
            7425.0, 7351.0, 5857.0, 5634.0, 5828.0, 6459.0, 4971.0, 4700.0, 4497.0, 4428.0, 4403.0,
            3531.0, 2988.0, 2173.0, 1650.0, 2838.0, 974.0, 756.0, 885.0, 1577.0, 199.0, 220.0, 0.0,
            3371.0, 2756.0, 1423.0, 2823.0, 852.0, 375.0,
        ],
        vec![
            9639.0, 9571.0, 6675.0, 7045.0, 7573.0, 8685.0, 5977.0, 6400.0, 5331.0, 7000.0, 7017.0,
            5734.0, 4547.0, 3238.0, 4779.0, 1244.0, 3064.0, 3674.0, 4064.0, 1900.0, 3222.0, 3583.0,
            3371.0, 0.0, 620.0, 1952.0, 560.0, 2580.0, 3173.0,
        ],
        vec![
            9230.0, 9160.0, 6466.0, 6741.0, 7222.0, 8268.0, 5719.0, 6037.0, 5084.0, 6514.0, 6525.0,
            5283.0, 4153.0, 2831.0, 4197.0, 1089.0, 2505.0, 3090.0, 3469.0, 1332.0, 2611.0, 2970.0,
            2756.0, 620.0, 0.0, 1334.0, 74.0, 1961.0, 2554.0,
        ],
        vec![
            8320.0, 8249.0, 6061.0, 6111.0, 6471.0, 7348.0, 5228.0, 5288.0, 4645.0, 5455.0, 5451.0,
            4335.0, 3400.0, 2164.0, 2931.0, 1643.0, 1368.0, 1834.0, 2177.0, 510.0, 1285.0, 1638.0,
            1423.0, 1952.0, 1334.0, 0.0, 1401.0, 648.0, 1231.0,
        ],
        vec![
            9300.0, 9231.0, 6523.0, 6805.0, 7289.0, 8338.0, 5780.0, 6105.0, 5143.0, 6587.0, 6598.0,
            5355.0, 4222.0, 2901.0, 4270.0, 1130.0, 2578.0, 3162.0, 3540.0, 1406.0, 2679.0, 3037.0,
            2823.0, 560.0, 74.0, 1401.0, 0.0, 2023.0, 2617.0,
        ],
        vec![
            8103.0, 8030.0, 6165.0, 6091.0, 6374.0, 7131.0, 5302.0, 5209.0, 4761.0, 5157.0, 5142.0,
            4143.0, 3376.0, 2285.0, 2470.0, 2252.0, 1208.0, 1439.0, 1699.0, 1002.0, 769.0, 1071.0,
            852.0, 2580.0, 1961.0, 648.0, 2023.0, 0.0, 594.0,
        ],
        vec![
            7799.0, 7725.0, 6164.0, 5977.0, 6187.0, 6832.0, 5281.0, 5052.0, 4787.0, 4802.0, 4776.0,
            3897.0, 3307.0, 2397.0, 2010.0, 2774.0, 1201.0, 1120.0, 1253.0, 1499.0, 440.0, 560.0,
            375.0, 3173.0, 2554.0, 1231.0, 2617.0, 594.0, 0.0,
        ],
    ];

    let graph_weights = Arc::new(graph_weights);
    let mut population = (0..NUMBER_OF_INDIVIDUALS_PER_POPULATION)
        .map(|_| TSP::new_with_random_path(graph_weights.clone()))
        .collect::<Vec<tsp::TSP>>();

    return population;
}
