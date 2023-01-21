﻿//using Tensorflow;
//using Tensorflow.Keras;
//using Tensorflow.Keras.ArgsDefinition;
//using Tensorflow.Keras.Engine;
//using Tensorflow.Keras.Layers;
//using Tensorflow.Keras.Models;
//using Tensorflow.NumPy;
//using static Tensorflow.Binding;
//using static Tensorflow.KerasApi;
//using Keras.Models;


using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.TensorExtensionMethods;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.torch.Tensor;
using static TorchSharp.TensorExtensionMethods;

using System.Collections;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using Rotinas_Iniciais_Simu;
using System.Reflection.Metadata.Ecma335;


var QL = new q_learning();


//Random rnd = new();
//int memory_size = Convert.ToInt32(5 * Math.Pow(10, 5));
//int batch_size = 32;
//float gamma = 1.0f;
//float exploration_max = 1.0f;
//float exploration_min = 0.01f;
//float exploration_decay = 0.9999f;
//float learning_rate = 0.003f;
//float learning_rate_Q = 0.3f;
//float tau = 0.125f;
//int n_outputs = 4;
//int n_inputs = 10;
//int layers = 2;
//int neurons = 32;
//int episodes = (int)1e5;
//var device = torch.cuda.is_available() ? torch.device("CUDA") : torch.device("cpu");

////dqn_torch model = new dqn_torch(batch_size, gamma, exploration_max, exploration_min, exploration_decay, learning_rate, tau, n_outputs, n_inputs, layers, neurons, "model", device);

////dqn_torch aux_model = new dqn_torch(batch_size, gamma, exploration_max, exploration_min, exploration_decay, learning_rate, tau, n_outputs, n_inputs, layers, neurons, "aux_model", device);

//var lin1 = Linear(n_inputs, neurons);
//var lin2 = Linear(neurons, neurons);
//var lin3 = Linear(neurons, n_outputs);

//var seq = new dqn_torch();
//var aux = new dqn_torch();
//var loss_func = MSELoss();
//Adam optimizer = new(seq.parameters(), learning_rate);

//simple_env env = new simple_env();
//replay_memory memory = new replay_memory(memory_size);

//List<int> list_rewards = new();
//List<float> list_losses = new();
//var ( state, done) = env.reset();
//var next_state = new float[,] { { state[0, 0], state[0, 1], state[0, 2], state[0, 3], state[0, 4], state[0, 5], state[0, 6], state[0, 7], state[0, 8], state[0, 9] } };
//int reward = 0;
//int action = -1;


//// ===================== LOOP DO DQN ==============================================
//for (int ep = 0; ep < episodes; ep++)
//{
//  (state, done) = env.reset();
//  List<int> r = new();
//  List<float> l = new();
//  while (!done)
//  {
//    if (ep % 100 == 0)
//    {
//      action = get_action(state, seq, rnd, true, true);
//    }
//    action = get_action(state, seq, rnd);

//    (next_state, reward, done) = env.step(state, action);
//    memory.remember(state, action, reward, next_state, done);

//    if (!(memory.size < batch_size + 100))
//    {

//      List<(float[,] state, int action, float reward, float[,] next_state, bool done)> samples = memory.sample(batch_size);

//      List<Tensor> Qs_current = new();
//      List<Tensor> Qs_expected = new();
//      foreach (var sample in samples)
//      {

//        //var stateT = tensor(new float[,] { { sample.state[0][0].item<float>(), sample.state[0][1].item<float>() } }, dtype: ScalarType.Float32, device: torch.CPU);
//        //var next_stateT = tensor(new float[,] { { sample.next_state[0][0].item<float>(), sample.next_state[0][1].item<float>() } }, dtype: ScalarType.Float32, device: torch.CPU);
//        //var actionT = tensor(new float[] { sample.action }, dtype: ScalarType.Float32, device: torch.CPU);
//        //var rewardT = tensor(new float[] { sample.reward }, dtype: ScalarType.Float32, device: torch.CPU);
//        //var doneT = tensor(new float[] { sample.done ? 1.0f : 0.0f }, dtype: ScalarType.Float32, device: torch.CPU);

//        var stateT = tensor(new float[,] { { sample.state[0, 0], sample.state[0, 1], sample.state[0, 2], sample.state[0, 3], sample.state[0, 4], sample.state[0, 5], sample.state[0, 6], sample.state[0, 7], sample.state[0, 8], sample.state[0, 9] } }, dtype: ScalarType.Float32, device: torch.CPU);
//        var next_stateT = tensor(new float[,] { { sample.next_state[0, 0], sample.next_state[0, 1], sample.next_state[0, 1], sample.next_state[0, 2], sample.next_state[0, 3], sample.next_state[0, 4], sample.next_state[0, 5], sample.next_state[0, 6], sample.next_state[0, 7], sample.next_state[0, 8], sample.next_state[0, 9] } }, dtype: ScalarType.Float32, device: torch.CPU);
//        var actionT = tensor(new float[] { sample.action }, dtype: ScalarType.Float32, device: torch.CPU);
//        var rewardT = tensor(new float[] { sample.reward }, dtype: ScalarType.Float32, device: torch.CPU);
//        var doneT = tensor(new float[] { sample.done ? 1.0f : 0.0f }, dtype: ScalarType.Float32, device: torch.CPU);
//        var done_float = sample.done ? 1.0f : 0.0f;

//        var Q_future = aux.forward(stateT).max().item<float>();
//        var Q_Expected = aux.forward(stateT);
//        Q_Expected[0][sample.action] = sample.reward + ((gamma * Q_future) * (1 - done_float));
//        var Q_current = seq.forward(stateT);

//        Qs_current.Add(Q_current);
//        Qs_expected.Add(Q_Expected);
//      }
//      //stateT.print(); actionT.print(); rewardT.print(); doneT.print();
//      //Q_Expected.print(); Q_current.print();

//      var Q_currentT = torch.cat(Qs_current);
//      var Q_expectedT = torch.cat(Qs_expected);

//      var loss = functional.mse_loss(Q_currentT, Q_expectedT, Reduction.Sum);
//      optimizer.zero_grad();
//      loss.backward();
//      optimizer.step();
//      l.Add(loss.item<float>());


//      //Tensor states = torch.cat(samples.Select(x => x.state).ToList());
//      //Tensor next_states = torch.cat(samples.Select(x => x.next_state.clone()).ToList());
//      //Tensor actions = torch.tensor(samples.Select(x => (long)x.action).ToList());
//      //Tensor rewards = torch.tensor(samples.Select(x => x.reward).ToList());
//      //Tensor dones = torch.tensor(samples.Select(x => x.done).ToList());

//      //Tensor expected = torch.zeros(new long[] { batch_size, 2 });
//      //Tensor pred = aux.forward(states);
//      //for (int i = 0; i < batch_size; i++)
//      //{

//      //  if (dones[i].item<bool>())
//      //  {
//      //    expected[i] = pred[i];
//      //    expected[i][actions[i]] = rewards[i];
//      //  }
//      //  else
//      //  {
//      //    expected[i] = pred[i];
//      //    var q_future = aux.forward(next_states[i]).max().item<float>();
//      //    expected[i][actions[i]] = rewards[i] + (gamma * q_future);
//      //  }
//      //}

//      //using var loss = loss_func.forward(seq.forward(states), expected);
//      //l.Add(loss.item<float>());

//      //seq.zero_grad();
//      //loss.backward();
//      //optimizer.step();

//      //pred.print();
//      //expected.print();
//      //loss.print();
//    }

//    state = new float[,] { { next_state[0, 0], next_state[0, 1], next_state[0, 2], next_state[0, 3], next_state[0, 4], next_state[0, 5], next_state[0, 6], next_state[0, 7], next_state[0, 8], next_state[0, 9] } };
//    r.Add(reward);
//  }
//  list_rewards.Add(r.Sum());
//  list_losses.Add(torch.mean(torch.tensor(l)).item<float>());

//  Console.WriteLine($" --- Episode {ep} --- Sum of Ep. Rewards: {list_rewards[ep]}, Mean Ep. Loss: {list_losses[ep]}, Exploration: {exploration_max}");


//  aux.load_state_dict(seq.state_dict());
//}

//float replay(replay_memory memory, dqn_torch model, Module<Tensor, Tensor> aux_model, optim.Optimizer optimizer)
//{

//  //for (int i=0; i < this.batch_size; i++)
//  //{
//  //  samples.Add(memory[rnd.Next(memory_size)]);
//  //}

//  // -=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=- APPROACH 1
//  //foreach ((var state, var action, var reward, var next_state, var done) in samples)
//  //{
//  //  var y_hat = model.forward(state);
//  //  var y = model.forward(state);
//  //  if (done)
//  //  {
//  //    y[0][action] = reward;
//  //  }
//  //  else
//  //  {
//  //    var Q_next = model.forward(next_state);
//  //    var Q_next_max = torch.max(Q_next);

//  //    y[0][action] = torch.tensor(reward) + (this.gamma * Q_next_max);
//  //    // Q_next.print(); Q_next_max.print();
//  //  }
//  //  var criterion = torch.nn.MSELoss(Reduction.Sum);
//  //  var output = criterion.forward(y_hat, y).to(device);
//  //  model.zero_grad();
//  //  output.backward();
//  //  optimizer.step();
//  //  //output.print(); y.print(); y_hat.print();
//  //  loss.Add(output.item<float>());
//  //}
//  //return torch.mean(torch.tensor(loss.ToArray())).item<float>();

//  // -=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=- APPROACH 2

//  if (memory.size < batch_size) return -1f;

//  var loss = new List<float>();
//  var _states = new float[32, 10];
//  var _targets = new float[32, 10];

//  List<(float[,] state, int action, float reward, float[,] next_state, bool done)> samples = memory.sample(batch_size);
//  var states = torch.cat(samples.Select(x => tensor(x.state)).ToList());

//  //for (int i = 0, max = samples.Count(); i < max; i++)
//  //{
//  //  _states[i, 0] = samples[i].state[0][0].item<float>();
//  //  _states[i, 1] = samples[i].state[0][1].item<float>();
//  //}
//  //var states = torch.tensor(_states);


//  Tensor targets;

//  targets = aux_model.forward(states);

//  for (int i = 0, imax = samples.Count(); i < imax; i++)
//  {
//    if (samples[i].done)
//    {
//      targets[i][samples[i].action] = samples[i].reward;
//    }
//    else
//    {
//      var next_state_q_values = aux_model.forward(tensor(samples[i].next_state));
//      var next_state_q_value = torch.max(next_state_q_values, 1).values.item<float>();
//      targets[i][samples[i].action] = samples[i].reward + (gamma * next_state_q_value);
//    }
//  }
//  model.train();
//  using (var d = torch.NewDisposeScope())
//  {
//    optimizer.zero_grad();
//    var prediction = model.forward(states);
//    var output = mse_loss(prediction, targets);
//    output.print();
//    output.backward();
//    optimizer.step();
//    return output.item<float>();
//  }
//}

//int get_action(float[,] state, dqn_torch nnet, Random rnd, bool should_explore = true, bool sould_print = false)
//{
//  int action = -1;
//  Tensor q_values;
//  if (should_explore)
//  {
//    exploration_max *= exploration_decay;
//    exploration_max = Math.Max(exploration_min, exploration_max);
//    if (torch.rand(1).item<float>() < exploration_max)
//      return rnd.Next(3);
//  }
//  using (torch.no_grad())
//  {
//    q_values = nnet.forward(tensor(state))[0];
//  }

//  var best_action = torch.argmax(q_values);

//  action = (int)best_action.item<long>();
//  Tensor q_values_init;
//  if (sould_print)
//  {

//    Console.WriteLine("Q VALUES OF INITIAL POSITION");
//    float[,] state_ini = new float[,] { { 10, 1, 1, 1, 1, 1, 1, 1, 1, 1 } };
//    q_values_init = nnet.forward(tensor(state_ini))[0];
//    Console.WriteLine("ESQUERDA:" + (float)q_values_init[0].item<float>());
//    Console.WriteLine("DIREITA: " + (float)q_values_init[1].item<float>());
//    Console.WriteLine("CIMA: " + (float)q_values_init[2].item<float>());
//    Console.WriteLine("BAIXO: " + (float)q_values_init[3].item<float>());
//    System.Threading.Thread.Sleep(10000);

//    float[,] state_2;
//    Console.WriteLine("Q VALUES OF NEXT STATE");
//    int reward_2;
//    bool done_2;
//    var best_action_2 = torch.argmax(q_values_init);
//    (state_2, reward_2, done_2) = env.step(state_ini, (int)best_action_2.item<long>());
//    q_values_init = nnet.forward(tensor(state_2))[0];
//    Console.WriteLine("ESQUERDA:" + (float)q_values_init[0].item<float>());
//    Console.WriteLine("DIREITA: " + (float)q_values_init[1].item<float>());
//    Console.WriteLine("CIMA: " + (float)q_values_init[2].item<float>());
//    Console.WriteLine("BAIXO: " + (float)q_values_init[3].item<float>());
//    System.Threading.Thread.Sleep(10000);


//  }
//  return action;
//}

//class dqn_torch : Module<Tensor, Tensor>
//{
//  //public int batch_size;
//  //public float gamma;
//  //public float exploration_max;
//  //public float exploration_min;
//  //public float exploration_decay;
//  //public float learning_rate;
//  //public float tau;
//  //public int n_inputs;
//  //public int n_actions;
//  //public int layers;
//  //public int neurons;

//  public Random rnd = new Random();
//  public Module<Tensor, Tensor> lin1 = Linear(10, 24);
//  public Module<Tensor, Tensor> lin2 = Linear(24, 24);
//  public Module<Tensor, Tensor> lin3 = Linear(24, 8);

//  public dqn_torch(
//                   //int batch_size,
//                   //float gamma,
//                   //float exploration_max,
//                   //float exploration_min,
//                   //float exploration_decay,
//                   //float learning_rate,
//                   //float tau,
//                   //int n_outputs,
//                   //int n_inputs,
//                   //int layers,
//                   //int neurons,
//                   //string name,
//                   torch.Device device = null) : base(nameof(dqn_torch))
//  {
//    RegisterComponents();

//    if (device != null && device.type == DeviceType.CUDA)
//      this.to(device);

//    //this.batch_size = batch_size;
//    //this.gamma = gamma;
//    //this.exploration_max = exploration_max;
//    //this.exploration_min = exploration_min;
//    //this.exploration_decay = exploration_decay;
//    //this.learning_rate = learning_rate;
//    //this.tau = tau;
//    //this.n_inputs = n_inputs;
//    //this.n_actions = n_outputs;
//    //this.neurons = neurons;
//    //this.layers = layers;
//  }

//  //public int get_action(Tensor state, bool should_explore = true)
//  //{
//  //  int action = -1;
//  //  Tensor q_values;
//  //  if (should_explore)
//  //  {
//  //    exploration_max *= exploration_decay;
//  //    exploration_max = Math.Max(exploration_min, exploration_max);
//  //    if (this.rnd.NextDouble() < exploration_max)
//  //      return rnd.Next(2);
//  //  }
//  //  using (torch.no_grad())
//  //  {
//  //    q_values = this.forward(state)[0];
//  //  }
//  //  var best_action = torch.argmax(q_values);

//  //  action = (int)best_action.item<long>();
//  //  return action;
//  //}

//  public override Tensor forward(Tensor input)
//  {
//    using var x1 = lin1.forward(input);
//    using var x2 = relu(x1);
//    using var x3 = lin2.forward(x2);
//    using var x4 = relu(x3);
//    return lin3.forward(x4);
//  }
//}









