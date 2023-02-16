//using Tensorflow;
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
using Python.Runtime;

Random rnd = new();
int memory_size = Convert.ToInt32(5 * Math.Pow(10, 5));
int batch_size = 32;
float gamma = 1.0f;
double exploration_max = 0.5;
double exploration_min = 0.01;
double exploration_decay = 1.0;
float learning_rate = 0.003f;
float learning_rate_Q = 1f;
float tau = 0.125f;
int n_outputs = 4;
int n_inputs = 10;
int layers = 2;
int neurons = 32;
int episodes = (int)1e6;
var device = torch.cuda.is_available() ? torch.device("CUDA") : torch.device("cpu");

//dqn_torch model = new dqn_torch(batch_size, gamma, exploration_max, exploration_min, exploration_decay, learning_rate, tau, n_outputs, n_inputs, layers, neurons, "model", device);

//dqn_torch aux_model = new dqn_torch(batch_size, gamma, exploration_max, exploration_min, exploration_decay, learning_rate, tau, n_outputs, n_inputs, layers, neurons, "aux_model", device);

var lin1 = Linear(n_inputs, neurons);
var lin2 = Linear(neurons, neurons);
var lin3 = Linear(neurons, n_outputs);

var seq = new dqn_torch();
var aux = new dqn_torch();
var loss_func = MSELoss();
Adam optimizer = new(seq.parameters(), learning_rate);

simple_env env = new simple_env();
replay_memory memory = new replay_memory(memory_size);

List<int> list_rewards = new();
List<float> list_losses = new();
(float[,] state, bool done) = env.reset();
var next_state = new float[,] { { state[0,0], state[0,1], state[0, 2], state[0, 3], state[0, 4], state[0, 5], state[0, 6], state[0, 7], state[0, 8], state[0, 9] } };
int reward = 0;
int action = -1;

float num;

//===================== LOOP DO Q LEARNING NORMAL COM INFO INCOMPLETA =======================
Console.WriteLine("Treinando o agente por meio do Q-Learning no ambiente com informacao totalmente incompleta -> Somente sua posicao eh conhecida");
float[,] qTableIncomplete = makeQtableIncomplete();
int max_steps;
for (int ep = 0; ep < episodes; ep++)
{
    Console.SetCursorPosition(0, Console.CursorTop);
    Console.Write( ep);

    Console.SetCursorPosition(10, Console.CursorTop);
    Console.Write(exploration_max);

    (state, done) = env.reset();
    max_steps = 0;
    while (!done && max_steps < 10)
    {
        action = get_actionQIncomplete(state, rnd, qTableIncomplete);
        max_steps++;

        (next_state, reward, done) = env.step(state, action);

        num = learning_rate_Q * (reward + gamma* MaxTableIncomplete(next_state, qTableIncomplete) - qTableIncomplete[(int)state[0, 0]-1, action]);
        qTableIncomplete[(int)state[0, 0]-1, action]
            = num + qTableIncomplete[(int)state[0, 0]-1, action];

        //Console.WriteLine($"{state[0,0]}, {next_state[0,0]}, {reward}, {done}, {action}");
        state = next_state;
    }
}

//===========================================================================================

// ============================= LOOP DE EXECUCAO DO Q LEARNING COM INFO INCOMPLETA ===================================
max_steps = 0;
Console.WriteLine("O treinamento acabou. Vamos ver como o agente se sai");
(state, done) = env.reset(false);
int rec_total_inc = 0;
while (!done && max_steps < 10)
{
    max_steps++;
    action = get_actionQIncomplete(state, rnd, qTableIncomplete, false);
    string direcao = "nada";
    if (action == 0)
    {
        direcao = "esquerda";
    }
    if (action == 1)
    {
        direcao = "direita";
    }
    if (action == 2)
    {
        direcao = "cima";
    }
    if (action == 3)
    {
        direcao = "baixo";
    }

    (next_state, reward, done) = env.step(state, action);
    rec_total_inc += reward;
    Console.WriteLine($"andei para a {direcao}.");
    Console.WriteLine("");
    state = next_state;
    Console.WriteLine(state[0, 2]);

}
Console.WriteLine($"Minha recompensa total foi de {rec_total_inc}");

// ================================================================================================


exploration_max = 1.0;
//====================== LOOP DO Q LEARNING NORMAL============================
Console.WriteLine("Treinando o agente por meio do Q-Learning no ambiente com informacao completa");
float[,,,,,,,,,,] qTable = makeQtable();
for (int ep = 0; ep < episodes; ep++)
{
    
    (state, done) = env.reset();
    
    while (!done)
    {
        //Console.WriteLine(action);
        //Console.WriteLine("");
        action = get_actionQ(state, rnd, qTable);
        

        (next_state, reward, done) = env.step(state, action);
        
        num = learning_rate_Q * (reward + gamma* MaxTable(next_state, qTable) - qTable[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], action]);
        qTable[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], action]
            = num + qTable[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], action];

        state = next_state;
    }
}
// ===========================================================================


// ============================= LOOP DE EXECUCAO DO Q LEARNING ===================================

Console.WriteLine("O treinamento acabou. Vamos ver como o agente se sai");
(state, done) = env.reset(false);
    int rec_total = 0;
    while (!done)
    {
        action = get_actionQ(state, rnd, qTable, false);
        string direcao = "nada";
        if (action == 0)
        {
            direcao = "esquerda";
        }
        if (action == 1)
        {
            direcao = "direita";
        }
        if (action == 2)
        {
            direcao = "cima";
        }
        if (action == 3)
        {
            direcao = "baixo";
        }
        
        (next_state, reward, done) = env.step(state, action);
        rec_total += reward;
        Console.WriteLine($"andei para a {direcao}.");
        Console.WriteLine("");
        state = next_state;
    Console.WriteLine(state[0, 2]);

    }
    Console.WriteLine($"Minha recompensa total foi de {rec_total}");

// ================================================================================================

//===================================== LOOP DO Q-LEARNING NO AMBIENTE INCOMPLETO COM 3 COMPONENTES DE ESTADO ====================
exploration_max = 1.0;
Console.WriteLine("Treinando o agente por meio do Q-Learning no ambiente com informacao incompleta -> O agente sabe sua posicao e a existencia dos diamantes azuis e amarelos");
float[,,,] qTableIncomplete_3 = makeQtableIncomplete_3();
simple_env_incomplete env_3 = new simple_env_incomplete();
for (int ep = 0; ep < episodes; ep++)
{

    (state, done) = env_3.reset();

    while (!done)
    {
        action = get_actionQIncomplete_3(state, rnd, qTableIncomplete_3);


        (next_state, reward, done) = env_3.step(state, action);

        num = learning_rate_Q * (reward + gamma* MaxTableIncomplete_3(next_state, qTableIncomplete_3) - qTableIncomplete_3[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], action]);
        qTableIncomplete_3[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], action]
            = num + qTableIncomplete_3[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], action];

        state = next_state;
    }
}

//================================================================================================================================


// ============================= LOOP DE EXECUCAO DO Q LEARNING NO AMBIENTE INCOMPLETO COM 3 COMPONENTES DE ESTADO ===================================

Console.WriteLine("O treinamento acabou. Vamos ver como o agente se sai");
(state, done) = env_3.reset();
int rec_total_3 = 0;
while (!done)
{
    action = get_actionQIncomplete_3(state, rnd, qTableIncomplete_3, false);
    string direcao = "nada";
    if (action == 0)
    {
        direcao = "esquerda";
    }
    if (action == 1)
    {
        direcao = "direita";
    }
    if (action == 2)
    {
        direcao = "cima";
    }
    if (action == 3)
    {
        direcao = "baixo";
    }

    (next_state, reward, done) = env_3.step(state, action);
    rec_total_3 += reward;
    Console.WriteLine($"andei para a {direcao}.");
    Console.WriteLine("");
    state = next_state;

}
Console.WriteLine($"Minha recompensa total foi de {rec_total_3}");

// ====================================================================================================================================

exploration_max = 1.0;
//===================================== LOOP DO Q-LEARNING NO AMBIENTE INCOMPLETO LANTERNA ====================

Console.WriteLine("Treinando o agente por meio do Q-Learning no ambiente com informacao incompleta LANTERNA -> O agente sabe sua posicao e as recompensas vizinhas");
float[,,,,,] qTableIncompleteLantern = makeQtableIncompleteLantern();
simple_env_lantern env_lantern = new simple_env_lantern();
for (int ep = 0; ep < episodes; ep++)
{

    (state, done) = env_lantern.reset();

    while (!done)
    {
        action = get_actionQIncompleteLantern(state, rnd, qTableIncompleteLantern);


        (next_state, reward, done) = env_lantern.step(state, action);

        num = learning_rate_Q * (reward + gamma* MaxTableIncompleteLantern(next_state, qTableIncompleteLantern) - qTableIncompleteLantern[(int)state[0, 0]-1, Trad_Lantern((int)state[0, 1]), Trad_Lantern((int)state[0, 2]), Trad_Lantern((int)state[0, 3]), Trad_Lantern((int)state[0, 4]), action]);
        qTableIncompleteLantern[(int)state[0, 0]-1, Trad_Lantern((int)state[0, 1]), Trad_Lantern((int)state[0, 2]), Trad_Lantern((int)state[0, 3]), Trad_Lantern((int)state[0, 4]), action]
            = num + qTableIncompleteLantern[(int)state[0, 0]-1, Trad_Lantern((int)state[0, 1]), Trad_Lantern((int)state[0, 2]), Trad_Lantern((int)state[0, 3]), Trad_Lantern((int)state[0, 4]), action];

        state = next_state;
    }
}

//================================================================================================================================

// ============================= LOOP DE EXECUCAO DO Q LEARNING NO AMBIENTE INCOMPLETO LANTERNA ===================================

Console.WriteLine("O treinamento acabou. Vamos ver como o agente se sai");
(state, done) = env_lantern.reset();
int rec_total_lantern = 0;
while (!done)
{
    action = get_actionQIncompleteLantern(state, rnd, qTableIncompleteLantern, false);
    string direcao = "nada";
    if (action == 0)
    {
        direcao = "esquerda";
    }
    if (action == 1)
    {
        direcao = "direita";
    }
    if (action == 2)
    {
        direcao = "cima";
    }
    if (action == 3)
    {
        direcao = "baixo";
    }

    (next_state, reward, done) = env_lantern.step(state, action);
    rec_total_lantern += reward;
    Console.WriteLine($"andei para a {direcao}.");
    Console.WriteLine("");
    state = next_state;

}
Console.WriteLine($"Minha recompensa total foi de {rec_total_lantern}");

// ====================================================================================================================================

// ===================== LOOP DO DQN ==============================================
for (int ep = 0; ep < episodes; ep++)
{
  (state, done) = env.reset();
  List<int> r = new();
  List<float> l = new();
  while (!done)
  {
    if (ep % 100 ==0)
        {
            action = get_action(state, seq, rnd, true, true);
        }
    action = get_action(state, seq, rnd);
    
    (next_state, reward, done) = env.step(state, action);
    memory.remember(state, action, reward, next_state, done);

    if (!(memory.size < batch_size+100))
    {

      List<(float[,] state, int action, float reward, float[,] next_state, bool done)> samples = memory.sample(batch_size);

      List<Tensor> Qs_current = new();
      List<Tensor> Qs_expected = new();
      foreach (var sample in samples)
      {
        
        //var stateT = tensor(new float[,] { { sample.state[0][0].item<float>(), sample.state[0][1].item<float>() } }, dtype: ScalarType.Float32, device: torch.CPU);
        //var next_stateT = tensor(new float[,] { { sample.next_state[0][0].item<float>(), sample.next_state[0][1].item<float>() } }, dtype: ScalarType.Float32, device: torch.CPU);
        //var actionT = tensor(new float[] { sample.action }, dtype: ScalarType.Float32, device: torch.CPU);
        //var rewardT = tensor(new float[] { sample.reward }, dtype: ScalarType.Float32, device: torch.CPU);
        //var doneT = tensor(new float[] { sample.done ? 1.0f : 0.0f }, dtype: ScalarType.Float32, device: torch.CPU);

        var stateT = tensor(new float[,] { {sample.state[0,0],sample.state[0,1],sample.state[0, 2],sample.state[0, 3],sample.state[0, 4],sample.state[0, 5],sample.state[0, 6],sample.state[0, 7],sample.state[0, 8],sample.state[0, 9] } }, dtype: ScalarType.Float32, device: torch.CPU);
        var next_stateT = tensor(new float[,] { {sample.next_state[0,0],sample.next_state[0,1],sample.next_state[0,1],sample.next_state[0,2],sample.next_state[0,3],sample.next_state[0,4],sample.next_state[0,5],sample.next_state[0, 6], sample.next_state[0, 7], sample.next_state[0, 8], sample.next_state[0, 9]} }, dtype: ScalarType.Float32, device: torch.CPU);
        var actionT = tensor(new float[] { sample.action }, dtype: ScalarType.Float32, device: torch.CPU);
        var rewardT = tensor(new float[] { sample.reward }, dtype: ScalarType.Float32, device: torch.CPU);
        var doneT = tensor(new float[] { sample.done ? 1.0f : 0.0f }, dtype: ScalarType.Float32, device: torch.CPU);
        var done_float = sample.done ? 1.0f : 0.0f;

        var Q_future = aux.forward(stateT).max().item<float>();
        var Q_Expected = aux.forward(stateT);
        Q_Expected[0][sample.action] = sample.reward + ((gamma * Q_future) * (1 - done_float));
        var Q_current = seq.forward(stateT);

        Qs_current.Add(Q_current);
        Qs_expected.Add(Q_Expected);
      }
      //stateT.print(); actionT.print(); rewardT.print(); doneT.print();
      //Q_Expected.print(); Q_current.print();

      var Q_currentT = torch.cat(Qs_current);
      var Q_expectedT = torch.cat(Qs_expected);

      var loss = functional.mse_loss(Q_currentT, Q_expectedT, Reduction.Sum);
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
      l.Add(loss.item<float>());
      

      //Tensor states = torch.cat(samples.Select(x => x.state).ToList());
      //Tensor next_states = torch.cat(samples.Select(x => x.next_state.clone()).ToList());
      //Tensor actions = torch.tensor(samples.Select(x => (long)x.action).ToList());
      //Tensor rewards = torch.tensor(samples.Select(x => x.reward).ToList());
      //Tensor dones = torch.tensor(samples.Select(x => x.done).ToList());

      //Tensor expected = torch.zeros(new long[] { batch_size, 2 });
      //Tensor pred = aux.forward(states);
      //for (int i = 0; i < batch_size; i++)
      //{

      //  if (dones[i].item<bool>())
      //  {
      //    expected[i] = pred[i];
      //    expected[i][actions[i]] = rewards[i];
      //  }
      //  else
      //  {
      //    expected[i] = pred[i];
      //    var q_future = aux.forward(next_states[i]).max().item<float>();
      //    expected[i][actions[i]] = rewards[i] + (gamma * q_future);
      //  }
      //}

      //using var loss = loss_func.forward(seq.forward(states), expected);
      //l.Add(loss.item<float>());

      //seq.zero_grad();
      //loss.backward();
      //optimizer.step();

      //pred.print();
      //expected.print();
      //loss.print();
    }

    state = new float[,] { { next_state[0,0], next_state[0,1], next_state[0, 2], next_state[0, 3], next_state[0, 4], next_state[0, 5], next_state[0, 6], next_state[0, 7], next_state[0, 8], next_state[0, 9] } };
    r.Add(reward);
  }
  list_rewards.Add(r.Sum());
  list_losses.Add(torch.mean(torch.tensor(l)).item<float>());

  Console.WriteLine($" --- Episode {ep} --- Sum of Ep. Rewards: {list_rewards[ep]}, Mean Ep. Loss: {list_losses[ep]}, Exploration: {exploration_max}");


    aux.load_state_dict(seq.state_dict());
}

float replay(replay_memory memory, dqn_torch model, Module<Tensor, Tensor> aux_model, optim.Optimizer optimizer)
{

  //for (int i=0; i < this.batch_size; i++)
  //{
  //  samples.Add(memory[rnd.Next(memory_size)]);
  //}

  // -=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=- APPROACH 1
  //foreach ((var state, var action, var reward, var next_state, var done) in samples)
  //{
  //  var y_hat = model.forward(state);
  //  var y = model.forward(state);
  //  if (done)
  //  {
  //    y[0][action] = reward;
  //  }
  //  else
  //  {
  //    var Q_next = model.forward(next_state);
  //    var Q_next_max = torch.max(Q_next);

  //    y[0][action] = torch.tensor(reward) + (this.gamma * Q_next_max);
  //    // Q_next.print(); Q_next_max.print();
  //  }
  //  var criterion = torch.nn.MSELoss(Reduction.Sum);
  //  var output = criterion.forward(y_hat, y).to(device);
  //  model.zero_grad();
  //  output.backward();
  //  optimizer.step();
  //  //output.print(); y.print(); y_hat.print();
  //  loss.Add(output.item<float>());
  //}
  //return torch.mean(torch.tensor(loss.ToArray())).item<float>();

  // -=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=- APPROACH 2

  if (memory.size < batch_size) return -1f;
  
  var loss = new List<float>();
  var _states = new float[32, 10];
  var _targets = new float[32, 10];
  
  List<(float[,] state, int action, float reward, float[,] next_state, bool done)> samples = memory.sample(batch_size);
  var states = torch.cat(samples.Select(x => tensor(x.state)).ToList());

  //for (int i = 0, max = samples.Count(); i < max; i++)
  //{
  //  _states[i, 0] = samples[i].state[0][0].item<float>();
  //  _states[i, 1] = samples[i].state[0][1].item<float>();
  //}
  //var states = torch.tensor(_states);

  
  Tensor targets;

  targets = aux_model.forward(states);

  for (int i = 0, imax = samples.Count(); i < imax; i++)
  {
    if (samples[i].done)
    {
      targets[i][samples[i].action] = samples[i].reward;
    }
    else
    {
      var next_state_q_values = aux_model.forward(tensor(samples[i].next_state));
      var next_state_q_value = torch.max(next_state_q_values, 1).values.item<float>();
      targets[i][samples[i].action] = samples[i].reward + (gamma * next_state_q_value);
    }
  }
  model.train();
  using (var d = torch.NewDisposeScope())
  {
    optimizer.zero_grad();
    var prediction = model.forward(states);
    var output = mse_loss(prediction, targets);
    output.print();
    output.backward();
    optimizer.step();
    return output.item<float>();
  }
}

int get_action(float[,] state, dqn_torch nnet, Random rnd, bool should_explore = true, bool sould_print=false)
{
  int action = -1;
  Tensor q_values;
  if (should_explore)
  {
    exploration_max *= exploration_decay;
    exploration_max = Math.Max(exploration_min, exploration_max);
    if (torch.rand(1).item<float>() < exploration_max)
      return rnd.Next(4);
  }
  using (torch.no_grad())
  {
    q_values = nnet.forward(tensor(state))[0];
  }
  
    var best_action = torch.argmax(q_values);

  action = (int)best_action.item<long>();
    Tensor q_values_init;
    if (sould_print)
    {

        Console.WriteLine("Q VALUES OF INITIAL POSITION");
        float[,] state_ini = new float[,] { { 10, 1, 1, 1, 1, 1, 1, 1, 1, 1 } };
        q_values_init = nnet.forward(tensor(state_ini))[0];
        Console.WriteLine("ESQUERDA:" + (float)q_values_init[0].item<float>());
        Console.WriteLine("DIREITA: " + (float)q_values_init[1].item<float>());
        Console.WriteLine("CIMA: " + (float)q_values_init[2].item<float>());
        Console.WriteLine("BAIXO: " + (float)q_values_init[3].item<float>());
        System.Threading.Thread.Sleep(10000);

        float[,] state_2;
        Console.WriteLine("Q VALUES OF NEXT STATE");
        int reward_2;
        bool done_2;
        var best_action_2 = torch.argmax(q_values_init);
        (state_2, reward_2, done_2) = env.step(state_ini, (int)best_action_2.item<long>());
        q_values_init = nnet.forward(tensor(state_2))[0];
        Console.WriteLine("ESQUERDA:" +(float)q_values_init[0].item<float>());
        Console.WriteLine("DIREITA: " +(float)q_values_init[1].item<float>());
        Console.WriteLine("CIMA: " +(float)q_values_init[2].item<float>());
        Console.WriteLine("BAIXO: " +(float)q_values_init[3].item<float>());
        System.Threading.Thread.Sleep(10000);


    }
  return action;
}


// ============================ GET ACTION DO Q LEARNING NORMAL ===================================================

 int get_actionQ(float[,] state, Random rnd, float[,,,,,,,,,,] qTable, bool should_explore = true, bool sould_print = false)
{
    float[] Q_state = new float[4];
    int action = -1;
    if (should_explore)
    {
        exploration_max *= exploration_decay;
        exploration_max = Math.Max(exploration_min, exploration_max);
        if (torch.rand(1).item<float>() < exploration_max)
        {
            Q_state[0] = qTable[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], 0];
            Q_state[1] = qTable[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], 1];
            Q_state[2] = qTable[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], 2];
            Q_state[3] = qTable[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], 3];
            List<int> randomUnvisited = new List<int>();

            for (int i = 0; i<4; i++)
            {
                if (Q_state[i] == 0)
                {
                    randomUnvisited.Add(i);
                }
            }

            if (randomUnvisited.Count == 0 || randomUnvisited.Count == 4)
            {
                return rnd.Next(4);
            }

            else
            {
                var randomIndex = rnd.Next(randomUnvisited.Count);
                return randomUnvisited[randomIndex];
            }
        }
    }

    int best_action = argMaxTable(state, qTable);

    return best_action;
}


int get_actionQIncomplete(float[,] state, Random rnd, float[,] qTable, bool should_explore = true, bool sould_print = false)
{
    float[] Q_state = new float[4];
    int action = -1;
    if (should_explore)
    {
        exploration_max *= exploration_decay;
        exploration_max = Math.Max(exploration_min, exploration_max);
        if (torch.rand(1).item<float>() < exploration_max)
        {
            Q_state[0] = qTable[(int)state[0, 0]-1, 0];
            Q_state[1] = qTable[(int)state[0, 0]-1, 1];
            Q_state[2] = qTable[(int)state[0, 0]-1, 2];
            Q_state[3] = qTable[(int)state[0, 0]-1, 3];
            List<int> randomUnvisited = new List<int>();

            for (int i = 0; i<4; i++)
            {
                if (Q_state[i] == 0)
                {
                    randomUnvisited.Add(i);
                }
            }

            if (randomUnvisited.Count == 0 || randomUnvisited.Count == 4)
            {
                return rnd.Next(4);
            }

            else
            {
                var randomIndex = rnd.Next(randomUnvisited.Count);
                return randomUnvisited[randomIndex];
            }
        }
    }

    int best_action = argMaxTableIncomplete(state, qTable);

    return best_action;
}

int get_actionQIncomplete_3(float[,] state, Random rnd, float[,,,] qTable, bool should_explore = true, bool sould_print = false)
{
    float[] Q_state = new float[4];
    int action = -1;
    if (should_explore)
    {
        exploration_max *= exploration_decay;
        exploration_max = Math.Max(exploration_min, exploration_max);
        if (torch.rand(1).item<float>() < exploration_max)
        {
            Q_state[0] = qTable[(int)state[0, 0]-1,(int)state[0,1],(int)state[0,2], 0];
            Q_state[1] = qTable[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], 1];
            Q_state[2] = qTable[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], 2];
            Q_state[3] = qTable[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], 3];
            List<int> randomUnvisited = new List<int>();

            for (int i = 0; i<4; i++)
            {
                if (Q_state[i] == 0)
                {
                    randomUnvisited.Add(i);
                }
            }

            if (randomUnvisited.Count == 0 || randomUnvisited.Count == 4)
            {
                return rnd.Next(4);
            }

            else
            {
                var randomIndex = rnd.Next(randomUnvisited.Count);
                return randomUnvisited[randomIndex];
            }
        }
    }

    int best_action = argMaxTableIncomplete_3(state, qTable);

    return best_action;
}


int get_actionQIncompleteLantern(float[,] state, Random rnd, float[,,,,,] qTable, bool should_explore = true, bool sould_print = false)
{
    float[] Q_state = new float[4];
    int action = -1;

    if (should_explore)
    {
        exploration_max *= exploration_decay;
        exploration_max = Math.Max(exploration_min, exploration_max);
        if (torch.rand(1).item<float>() < exploration_max)
        {
            Q_state[0] = qTable[(int)state[0, 0]-1, Trad_Lantern((int)state[0, 1]), Trad_Lantern((int)state[0, 2]), Trad_Lantern((int)state[0, 3]), Trad_Lantern((int)state[0, 4]), 0];
            Q_state[1] = qTable[(int)state[0, 0]-1, Trad_Lantern((int)state[0, 1]), Trad_Lantern((int)state[0, 2]), Trad_Lantern((int)state[0, 3]), Trad_Lantern((int)state[0, 4]), 1];
            Q_state[2] = qTable[(int)state[0, 0]-1, Trad_Lantern((int)state[0, 1]), Trad_Lantern((int)state[0, 2]), Trad_Lantern((int)state[0, 3]), Trad_Lantern((int)state[0, 4]), 2];
            Q_state[3] = qTable[(int)state[0, 0]-1, Trad_Lantern((int)state[0, 1]), Trad_Lantern((int)state[0, 2]), Trad_Lantern((int)state[0, 3]), Trad_Lantern((int)state[0, 4]), 3];
            List<int> randomUnvisited = new List<int>();

            for (int i = 0; i<4; i++)
            {
                if (Q_state[i] == 0)
                {
                    randomUnvisited.Add(i);
                }
            }

            if (randomUnvisited.Count == 0 || randomUnvisited.Count == 4)
            {
                return rnd.Next(4);
            }

            else
            {
                var randomIndex = rnd.Next(randomUnvisited.Count);
                return randomUnvisited[randomIndex];
            }
        }
    }

    int best_action = argMaxTableIncompleteLantern(state, qTable);

    return best_action;
}

// ===============================================================================================================


int argMaxTable(float[,]  state, float[,,,,,,,,,,] qTable)
{
    float max = -8000;
    float compare;
    int action = -1;
    for (int i = 0; i < 4; i++)
    {
        compare = qTable[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], i];
        if (compare> max)
        {
            max = compare;
            action = i;
        }

        
    }
    return action;
}

int argMaxTableIncomplete(float[,] state, float[,] qTable)
{
    float max = -8000;
    float compare;
    int action = -1;
    for (int i = 0; i < 4; i++)
    {
        compare = qTable[(int)state[0, 0]-1,  i];
        if (compare> max)
        {
            max = compare;
            action = i;
        }


    }
    return action;
}


int argMaxTableIncomplete_3(float[,] state, float[,,,] qTable)
{
    float max = -8000;
    float compare;
    int action = -1;
    for (int i = 0; i < 4; i++)
    {
        compare = qTable[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], i];
        if (compare> max)
        {
            max = compare;
            action = i;
        }


    }
    return action;
}

int argMaxTableIncompleteLantern(float[,] state, float[,,,,,] qTable)
{
    float max = -8000;
    float compare;
    int action = -1;
    for (int i = 0; i < 4; i++)
    {
        compare = qTable[(int)state[0, 0]-1, Trad_Lantern((int)state[0, 1]), Trad_Lantern((int)state[0, 2]), Trad_Lantern((int)state[0, 3]), Trad_Lantern((int)state[0, 4]), i];
        if (compare> max)
        {
            max = compare;
            action = i;
        }


    }
    return action;
}


float[,,,,,,,,,,] makeQtable()
    {
        return new float[16, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4];
    }

float[,] makeQtableIncomplete()
{
    return new float[16, 4];
}

float[,,,] makeQtableIncomplete_3()
{
    return new float[16,2,2,4];
}

float[,,,,,] makeQtableIncompleteLantern()
{
    return new float[16, 6, 6, 6, 6, 4];
}


float MaxTable(float[,] state, float[,,,,,,,,,,] qTable)
{
    float max = -8000;
    float compare;
    for (int i = 0; i < 4; i++)
    {
        compare = qTable[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], i];
        if (compare> max)
        {
            max = compare;
        }


    }
    return max;
}

float MaxTableIncomplete(float[,] state, float[,] qTable)
{
    float max = -8000;
    float compare;
    for (int i = 0; i < 4; i++)
    {
        compare = qTable[(int)state[0, 0]-1, i];
        if (compare> max)
        {
            max = compare;
        }


    }
    return max;
}

float MaxTableIncomplete_3(float[,] state, float[,,,] qTable)
{
    float max = -8000;
    float compare;
    for (int i = 0; i < 4; i++)
    {
        compare = qTable[(int)state[0, 0]-1, (int)state[0, 1], (int)state[0, 2], i];
        if (compare> max)
        {
            max = compare;
        }


    }
    return max;
}

float MaxTableIncompleteLantern(float[,] state, float[,,,,,] qTable)
{
    float max = -8000;
    float compare;
    for (int i = 0; i < 4; i++)
    {
        compare = qTable[(int)state[0, 0]-1, Trad_Lantern((int)state[0, 1]), Trad_Lantern((int)state[0, 2]), Trad_Lantern((int)state[0, 3]), Trad_Lantern((int)state[0, 4]), i];
        if (compare> max)
        {
            max = compare;
        }


    }
    return max;
}


int Trad_Lantern(int reward)
{
    if (reward == -2)
    {
        return 0;
    }
    if (reward == -1)
    {
        return 1;
    }
    if (reward == 2)
    {
        return 2;
    }
    if (reward == 3)
    {
        return 3;
    }
    if (reward == 5)
    {
        return 4;
    }
    else
    {
        return 5;
    }
}


class dqn_torch : Module<Tensor, Tensor>
{
  //public int batch_size;
  //public float gamma;
  //public float exploration_max;
  //public float exploration_min;
  //public float exploration_decay;
  //public float learning_rate;
  //public float tau;
  //public int n_inputs;
  //public int n_actions;
  //public int layers;
  //public int neurons;

  public Random rnd = new Random();
  public Module<Tensor, Tensor> lin1 = Linear(10, 24);
  public Module<Tensor, Tensor> lin2 = Linear(24, 24);
  public Module<Tensor, Tensor> lin3 = Linear(24, 8);

  public dqn_torch(
                   //int batch_size,
                   //float gamma,
                   //float exploration_max,
                   //float exploration_min,
                   //float exploration_decay,
                   //float learning_rate,
                   //float tau,
                   //int n_outputs,
                   //int n_inputs,
                   //int layers,
                   //int neurons,
                   //string name,
                   torch.Device device = null) : base(nameof(dqn_torch))
  {
    RegisterComponents();

    if (device != null && device.type == DeviceType.CUDA)
      this.to(device);

    //this.batch_size = batch_size;
    //this.gamma = gamma;
    //this.exploration_max = exploration_max;
    //this.exploration_min = exploration_min;
    //this.exploration_decay = exploration_decay;
    //this.learning_rate = learning_rate;
    //this.tau = tau;
    //this.n_inputs = n_inputs;
    //this.n_actions = n_outputs;
    //this.neurons = neurons;
    //this.layers = layers;
  }

  //public int get_action(Tensor state, bool should_explore = true)
  //{
  //  int action = -1;
  //  Tensor q_values;
  //  if (should_explore)
  //  {
  //    exploration_max *= exploration_decay;
  //    exploration_max = Math.Max(exploration_min, exploration_max);
  //    if (this.rnd.NextDouble() < exploration_max)
  //      return rnd.Next(2);
  //  }
  //  using (torch.no_grad())
  //  {
  //    q_values = this.forward(state)[0];
  //  }
  //  var best_action = torch.argmax(q_values);

  //  action = (int)best_action.item<long>();
  //  return action;
  //}

  public override Tensor forward(Tensor input)
  {
    using var x1 = lin1.forward(input);
    using var x2 = relu (x1);
    using var x3 = lin2.forward(x2);
    using var x4 = relu(x3);
    return lin3.forward(x4);
  }
}

public class simple_env
{

    private int[,] _rewardMatrix = { { -1, 10, 2, 5}, { -1, 3, 5, 5}, { -1, -1, -2, 5}, { -1, -1, -2, 5} };

    public int[,] RewardMatrix
    {
        get { return _rewardMatrix; }
        set { _rewardMatrix = value; }
    }
    public int calc_reward(float[,] next_state)

  {
        int row = ((int)next_state[0, 0]-1)/4;
        int col = ((int)next_state[0, 0]-1)%4;
        return this.RewardMatrix[row, col];
        
    }

  public (float[,], bool) reset(bool train=true)
  {
        Random rnd = new Random();
        int rand_pos = 2;
        var done = false;
        float[,] state;
        if (train)
        {
            state = new float[,] { { 10, 1, 1, 1, 1, 1, 1, 1, 1, 1 } };
            while (rand_pos ==2 || rand_pos ==13) {
                rand_pos = rnd.Next(1, 17);
                state = new float[,] { { rand_pos, 1, 1, 1, 1, 1, 1, 1, 1, 1 } };

            }
            this.RewardMatrix[0, 0] = -1;
            this.RewardMatrix[0, 1] = 10;
            this.RewardMatrix[0, 2] = 2;
            this.RewardMatrix[0, 3] = 5;
            this.RewardMatrix[1, 0] = -1;
            this.RewardMatrix[1, 1] = 3;
            this.RewardMatrix[1, 2] = 5;
            this.RewardMatrix[1, 3] = 5;
            this.RewardMatrix[2, 0] = -1;
            this.RewardMatrix[2, 1] = -1;
            this.RewardMatrix[2, 2] = -2;
            this.RewardMatrix[2, 3] = 5;
            this.RewardMatrix[3, 0] = -1;
            this.RewardMatrix[3, 1] = -1;
            this.RewardMatrix[3, 2] = -2;
            this.RewardMatrix[3, 3] = 5;
            return (state, done);
        }

        else
        {
            state = new float[,] { { 10, 1, 1, 1, 1, 1, 1, 1, 1, 1 } };
            this.RewardMatrix[0, 0] = -1;
            this.RewardMatrix[0, 1] = 10;
            this.RewardMatrix[0, 2] = 2;
            this.RewardMatrix[0, 3] = 5;
            this.RewardMatrix[1, 0] = -1;
            this.RewardMatrix[1, 1] = 3;
            this.RewardMatrix[1, 2] = 5;
            this.RewardMatrix[1, 3] = 5;
            this.RewardMatrix[2, 0] = -1;
            this.RewardMatrix[2, 1] = -1;
            this.RewardMatrix[2, 2] = -2;
            this.RewardMatrix[2, 3] = 5;
            this.RewardMatrix[3, 0] = -1;
            this.RewardMatrix[3, 1] = -1;
            this.RewardMatrix[3, 2] = -2;
            this.RewardMatrix[3, 3] = 5;
            return (state, done);
        }

    }

    

    public (float[,], int, bool) step(float[,] state, int action)
  {
    int reward = 0;
    float[,] next_state = new float[,] { { state[0,0], state[0,1], state[0,2], state[0, 3], state[0, 4], state[0, 5], state[0, 6], state[0, 7], state[0, 8], state[0, 9] } };
    //var _state = new List<float>();
    //var _next_state = new List<float>();
    var done = false;

        //for (long i = 0, imax = state.size()[0] + 1; i < imax; i++)
        //{
        //  _state.Add(state[0][i].item<float>());
        //  _next_state.Add((float)state[0][i].item<float>());
        //}

        if (action == 0) // esquerda
        {
            //Console.WriteLine("andei pra esquerda");
            if (state[0,0] == 1)
            {
                next_state[0,0] = 4;
            }

            else if (state[0,0] == 3)
            {

                next_state[0,0] = 2;
                done = true;
            }

            else
            {
                next_state[0,0] -= 1;
            }

            reward = calc_reward(next_state);

        }



        else if (action == 1) // direita 
        {
            //Console.WriteLine("andei pra direita");
            if (state[0,0] == 16)
            {
                next_state[0,0] = 13;
            }

            else if (state[0,0] == 1)
            {
                next_state[0,0] = 2;
                done = true;
            }

            else
            {
                next_state[0,0] += 1;
            }
            reward = calc_reward(next_state);

        }

        else if (action == 2) // cima 
        {
            //Console.WriteLine("andei pra cima");
            if (state[0,0] == 1)
            {
                next_state[0,0] = 13;
            }

            else if (state[0,0] == 3)
            {
                next_state[0,0] = 15;
            }

            else if (state[0,0] == 4)
            {
                next_state[0,0] = 16;
            }

            else if (state[0,0] == 6)
            {
                next_state[0,0] = 2;
                done = true;
            }

            else
            {
                next_state[0,0] -= 4;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 3) // baixo
        {
            //Console.WriteLine("andei pra baixo");
            if (state[0,0] == 13)
            {
                next_state[0,0] = 1;
            }

            else if (state[0,0] == 14)
            {
                next_state[0,0] = 2;
                done=true;
            }

            else if (state[0,0] == 15)
            {
                next_state[0,0] = 3;
            }

            else if (state[0,0] == 16)
            {
                next_state[0,0] = 4;
            }

            else
            {
                next_state[0,0] += 4;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 4) // diagonal cima direita
        {
            //Console.WriteLine("andei pra diagonal cima direita");
            if (state[0,0] == 1)
            {
                next_state[0,0] = 14;
            }

            else if (state[0,0] == 3)
            {
                next_state[0,0] = 16;
            }


            else if (state[0,0] == 5)
            {
                next_state[0,0] = 2;
                done = true;
            }

            else
            {
                next_state[0,0] -= 3;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 5) // diagonal cima esquerda
        {
            if (state[0,0] == 5)
            {
                next_state[0,0] = 4;
            }

            else if (state[0,0] == 4)
            {
                next_state[0,0] = 15;
            }

            else if (state[0,0] == 3)
            {
                next_state[0,0] = 14;
            }

            else if (state[0,0] == 1)
            {
                next_state[0,0] = 16;
            }

            else if (state[0,0] == 7)
            {
                next_state[0,0] = 2;
                done=true;
            }

            else
            {
                next_state[0,0] -= 5;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 6) // diagonal baixo esquerda
        {

            if (state[0,0] == 14)
            {
                next_state[0,0] = 1;
            }

            else if (state[0,0] == 15)
            {
                next_state[0,0] = 2;
                done = true;
            }

            else if (state[0,0] == 16)
            {
                next_state[0,0] = 3;
            }

            else
            {
                next_state[0,0] += 3;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 7) // diagonal baixo direita
        {
            if (state[0,0] == 12)
            {
                next_state[0,0] = 13;
            }

            else if (state[0,0] == 13)
            {
                next_state[0,0] = 2;
                done = true;
            }

            else if (state[0,0] == 14)
            {
                next_state[0,0] = 3;
            }

            else if (state[0,0] == 15)
            {
                next_state[0,0] = 4;
            }

            else if (state[0,0] == 16)
            {
                next_state[0,0] = 1;
            }

            else
            {
                next_state[0,0] += 5;
            }
            reward = calc_reward(next_state);
        }

        if (next_state[0,0] == 2 || next_state[0,0] == 13)
        {
            done = true;
            //Console.WriteLine("caiu no fogo ou ganhou");
        }
        if (next_state[0,1] == 1 && next_state[0,0] == 3)
        {
            next_state[0,1] = 0;
            this.RewardMatrix[0, 2] = -1;
            next_state[0,4] = 0;
            next_state[0,5] = 0;
            next_state[0,6] = 0;
            next_state[0,7] = 0;
            this.RewardMatrix[0, 3] = -1;
            this.RewardMatrix[1, 3] = -1;
            this.RewardMatrix[2, 3] = -1;
            this.RewardMatrix[3, 3] = -1;
            //Console.WriteLine("diamante parou de existir");
        }

        if (next_state[0,2] == 1 && next_state[0,0] == 6)
        {
            next_state[0,2] = 0;
            this.RewardMatrix[1, 1] = -1;
            next_state[0,4] = 0;
            next_state[0,5] = 0;
            next_state[0,6] = 0;
            next_state[0,7] = 0;
            this.RewardMatrix[0, 3] = -1;
            this.RewardMatrix[1, 3] = -1;
            this.RewardMatrix[2, 3] = -1;
            this.RewardMatrix[3, 3] = -1;
            //Console.WriteLine("diamante parou de existir");
        }

        if (next_state[0,3] == 1 && next_state[0,0] == 7)
        {
            next_state[0,3] = 0;
            this.RewardMatrix[1, 2] = -1;
            next_state[0,4] = 0;
            next_state[0,5] = 0;
            next_state[0,6] = 0;
            next_state[0,7] = 0;
            this.RewardMatrix[0, 3] = -1;
            this.RewardMatrix[1, 3] = -1;
            this.RewardMatrix[2, 3] = -1;
            this.RewardMatrix[3, 3] = -1;
            //Console.WriteLine("diamante parou de existir");
        }

        if (next_state[0,4] == 1 && next_state[0,0] == 16)
        {
            this.RewardMatrix[3, 3] = -1;
            next_state[0,4] = 0;
            next_state[0,1] = 0;
            next_state[0,2] = 0;
            next_state[0,3] = 0;
            this.RewardMatrix[0, 2] = -1;
            this.RewardMatrix[1, 1] = -1;
            this.RewardMatrix[1, 2] = -1;
            //Console.WriteLine("diamante parou de existir");
        }

        if (next_state[0,5] == 1 && next_state[0,0] == 12)
        {
            this.RewardMatrix[2, 3] = -1;
            next_state[0,5] = 0;
            next_state[0,1] = 0;
            next_state[0,2] = 0;
            next_state[0,3] = 0;
            this.RewardMatrix[0, 2] = -1;
            this.RewardMatrix[1, 1] = -1;
            this.RewardMatrix[1, 2] = -1;
            //Console.WriteLine("diamante parou de existir");
        }

        if (next_state[0,6] == 1 && next_state[0,0] == 8)
        {
            this.RewardMatrix[1, 3] = -1;
            next_state[0,6] = 0;
            next_state[0,1] = 0;
            next_state[0,2] = 0;
            next_state[0,3] = 0;
            this.RewardMatrix[0, 2] = -1;
            this.RewardMatrix[1, 1] = -1;
            this.RewardMatrix[1, 2] = -1;
            //Console.WriteLine("diamante parou de existir");
        }

        if (next_state[0,7] == 1 && next_state[0,0] == 4)
        {
            this.RewardMatrix[0, 3] = -1;
            next_state[0,7] = 0;
            next_state[0,1] = 0;
            next_state[0,2] = 0;
            next_state[0,3] = 0;
            this.RewardMatrix[0, 2] = -1;
            this.RewardMatrix[1, 1] = -1;
            this.RewardMatrix[1, 2] = -1;
            //Console.WriteLine("diamante parou de existir");
        }

        if (next_state[0,8] == 1 && next_state[0,0] == 11)
        {
            next_state[0,8] = 0;
            //Console.WriteLine("diamante parou de existir");
            this.RewardMatrix[2, 2] = -1;
        }

        if (next_state[0,9] == 1 && next_state[0,0] == 15)
        {
            next_state[0,9] = 0;
            this.RewardMatrix[3, 2] = -1;
            //Console.WriteLine("diamante parou de existir");
        }

        return (next_state, reward, done);
    }

}
public class simple_env_incomplete
{

    private int[,] _rewardMatrix = { { -1, 10, 2, 5 }, { -1, 3, 5, 5 }, { -1, -1, -2, 5 }, { -1, -1, -2, 5 } };

    public int[,] RewardMatrix
    {
        get { return _rewardMatrix; }
        set { _rewardMatrix = value; }
    }
    public int calc_reward(float[,] next_state)

    {
        int row = ((int)next_state[0, 0]-1)/4;
        int col = ((int)next_state[0, 0]-1)%4;
        return this.RewardMatrix[row, col];

    }

    public (float[,], bool) reset(bool train=true)
    {
        Random rnd = new Random();
        int rand_pos = 2;
        var done = false;
        float[,] state;
        if (train)
        {
            state = new float[,] { { 10, 1, 1 } };
            while (rand_pos ==2 || rand_pos ==13)
            {
                rand_pos = rnd.Next(1, 17);
                state = new float[,] { { rand_pos, 1, 1, 1, 1, 1, 1, 1, 1, 1 } };

            }
            this.RewardMatrix[0, 0] = -1;
            this.RewardMatrix[0, 1] = 10;
            this.RewardMatrix[0, 2] = 2;
            this.RewardMatrix[0, 3] = 5;
            this.RewardMatrix[1, 0] = -1;
            this.RewardMatrix[1, 1] = 3;
            this.RewardMatrix[1, 2] = 5;
            this.RewardMatrix[1, 3] = 5;
            this.RewardMatrix[2, 0] = -1;
            this.RewardMatrix[2, 1] = -1;
            this.RewardMatrix[2, 2] = -2;
            this.RewardMatrix[2, 3] = 5;
            this.RewardMatrix[3, 0] = -1;
            this.RewardMatrix[3, 1] = -1;
            this.RewardMatrix[3, 2] = -2;
            this.RewardMatrix[3, 3] = 5;
            return (state, done);
        }

        else
        {
            state = new float[,] { { 10, 1, 1 } };
            this.RewardMatrix[0, 0] = -1;
            this.RewardMatrix[0, 1] = 10;
            this.RewardMatrix[0, 2] = 2;
            this.RewardMatrix[0, 3] = 5;
            this.RewardMatrix[1, 0] = -1;
            this.RewardMatrix[1, 1] = 3;
            this.RewardMatrix[1, 2] = 5;
            this.RewardMatrix[1, 3] = 5;
            this.RewardMatrix[2, 0] = -1;
            this.RewardMatrix[2, 1] = -1;
            this.RewardMatrix[2, 2] = -2;
            this.RewardMatrix[2, 3] = 5;
            this.RewardMatrix[3, 0] = -1;
            this.RewardMatrix[3, 1] = -1;
            this.RewardMatrix[3, 2] = -2;
            this.RewardMatrix[3, 3] = 5;
            return (state, done);
        }

    }



    public (float[,], int, bool) step(float[,] state, int action)
    {
        int reward = 0;
        float[,] next_state = new float[,] { { state[0, 0], state[0, 1], state[0, 2] } };
        var done = false;


        if (action == 0) // esquerda
        {
            if (state[0, 0] == 1)
            {
                next_state[0, 0] = 4;
            }

            else if (state[0, 0] == 3)
            {

                next_state[0, 0] = 2;
                done = true;
            }

            else
            {
                next_state[0, 0] -= 1;
            }

            reward = calc_reward(next_state);

        }



        else if (action == 1) // direita 
        {
            //Console.WriteLine("andei pra direita");
            if (state[0, 0] == 16)
            {
                next_state[0, 0] = 13;
            }

            else if (state[0, 0] == 1)
            {
                next_state[0, 0] = 2;
                done = true;
            }

            else
            {
                next_state[0, 0] += 1;
            }
            reward = calc_reward(next_state);

        }

        else if (action == 2) // cima 
        {
            //Console.WriteLine("andei pra cima");
            if (state[0, 0] == 1)
            {
                next_state[0, 0] = 1;
            }

            else if (state[0, 0] == 3)
            {
                next_state[0, 0] = 3;
            }

            else if (state[0, 0] == 4)
            {
                next_state[0, 0] = 4;
            }

            else if (state[0, 0] == 6)
            {
                next_state[0, 0] = 2;
                done = true;
            }

            else
            {
                next_state[0, 0] -= 4;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 3) // baixo
        {
            //Console.WriteLine("andei pra baixo");
            if (state[0, 0] == 13)
            {
                next_state[0, 0] = 1;
            }

            else if (state[0, 0] == 14)
            {
                next_state[0, 0] = 2;
                done=true;
            }

            else if (state[0, 0] == 15)
            {
                next_state[0, 0] = 3;
            }

            else if (state[0, 0] == 16)
            {
                next_state[0, 0] = 4;
            }

            else
            {
                next_state[0, 0] += 4;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 4) // diagonal cima direita
        {
            //Console.WriteLine("andei pra diagonal cima direita");
            if (state[0, 0] == 1)
            {
                next_state[0, 0] = 14;
            }

            else if (state[0, 0] == 3)
            {
                next_state[0, 0] = 16;
            }


            else if (state[0, 0] == 5)
            {
                next_state[0, 0] = 2;
                done = true;
            }

            else
            {
                next_state[0, 0] -= 3;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 5) // diagonal cima esquerda
        {
            if (state[0, 0] == 5)
            {
                next_state[0, 0] = 4;
            }

            else if (state[0, 0] == 4)
            {
                next_state[0, 0] = 15;
            }

            else if (state[0, 0] == 3)
            {
                next_state[0, 0] = 14;
            }

            else if (state[0, 0] == 1)
            {
                next_state[0, 0] = 16;
            }

            else if (state[0, 0] == 7)
            {
                next_state[0, 0] = 2;
                done=true;
            }

            else
            {
                next_state[0, 0] -= 5;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 6) // diagonal baixo esquerda
        {

            if (state[0, 0] == 14)
            {
                next_state[0, 0] = 1;
            }

            else if (state[0, 0] == 15)
            {
                next_state[0, 0] = 2;
                done = true;
            }

            else if (state[0, 0] == 16)
            {
                next_state[0, 0] = 3;
            }

            else
            {
                next_state[0, 0] += 3;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 7) // diagonal baixo direita
        {
            if (state[0, 0] == 12)
            {
                next_state[0, 0] = 13;
            }

            else if (state[0, 0] == 13)
            {
                next_state[0, 0] = 2;
                done = true;
            }

            else if (state[0, 0] == 14)
            {
                next_state[0, 0] = 3;
            }

            else if (state[0, 0] == 15)
            {
                next_state[0, 0] = 4;
            }

            else if (state[0, 0] == 16)
            {
                next_state[0, 0] = 1;
            }

            else
            {
                next_state[0, 0] += 5;
            }
            reward = calc_reward(next_state);
        }

        if (next_state[0, 0] == 2 || next_state[0, 0] == 13)
        {
            done = true;
            //Console.WriteLine("caiu no fogo ou ganhou");
        }

        

        if (this.RewardMatrix[0,2]==2 && next_state[0, 0] == 3)
        {
            this.RewardMatrix[0, 2] = -1;
            next_state[0, 2] = 0;
            this.RewardMatrix[0, 3] = -1;
            this.RewardMatrix[1, 3] = -1;
            this.RewardMatrix[2, 3] = -1;
            this.RewardMatrix[3, 3] = -1;
            //Console.WriteLine("diamante da pos 3 parou de existir");
            //Console.WriteLine("diamantes amarelos pararam de existir");
        }

        if (this.RewardMatrix[1,1] == 3 && next_state[0, 0] == 6)
        {
            this.RewardMatrix[1, 1] = -1;
            next_state[0, 2] = 0;
            this.RewardMatrix[0, 3] = -1;
            this.RewardMatrix[1, 3] = -1;
            this.RewardMatrix[2, 3] = -1;
            this.RewardMatrix[3, 3] = -1;
            //Console.WriteLine("diamante da pos 6 parou de existir");
            //Console.WriteLine("diamantes amarelos pararam de existir");
        }

        if (this.RewardMatrix[1,2] == 5 && next_state[0, 0] == 7)
        {
            this.RewardMatrix[1, 2] = -1;
            next_state[0, 2] = 0;
            this.RewardMatrix[0, 3] = -1;
            this.RewardMatrix[1, 3] = -1;
            this.RewardMatrix[2, 3] = -1;
            this.RewardMatrix[3, 3] = -1;
            //Console.WriteLine("diamante da pos 7 parou de existir");
            //Console.WriteLine("diamantes amarelos pararam de existir");
        }

        if (this.RewardMatrix[3,3]==5 && next_state[0, 0] == 16)
        {
            this.RewardMatrix[3, 3] = -1;
            next_state[0, 1] = 0;
            this.RewardMatrix[0, 2] = -1;
            this.RewardMatrix[1, 1] = -1;
            this.RewardMatrix[1, 2] = -1;
            //Console.WriteLine("diamante da pos 16 parou de existir");
            //Console.WriteLine("diamantes azuis pararam de existir");
        }

        if (this.RewardMatrix[2,3] == 5 && next_state[0, 0] == 12)
        {
            this.RewardMatrix[2, 3] = -1;
            next_state[0, 1] = 0;
            this.RewardMatrix[0, 2] = -1;
            this.RewardMatrix[1, 1] = -1;
            this.RewardMatrix[1, 2] = -1;
            //Console.WriteLine("diamante da pos 12 parou de existir");
            //Console.WriteLine("diamantes azuis pararam de existir");
        }

        if (this.RewardMatrix[1,3]==5 && next_state[0, 0] == 8)
        {
            this.RewardMatrix[1, 3] = -1;
            next_state[0, 1] = 0;
            this.RewardMatrix[0, 2] = -1;
            this.RewardMatrix[1, 1] = -1;
            this.RewardMatrix[1, 2] = -1;
            //Console.WriteLine("diamante da pos 8 parou de existir");
            //Console.WriteLine("diamantes azuis pararam de existir");
        }

        if (this.RewardMatrix[0, 3] == 5 && next_state[0, 0] == 4)
        {
            this.RewardMatrix[0, 3] = -1;
            next_state[0, 1] = 0;
            this.RewardMatrix[0, 2] = -1;
            this.RewardMatrix[1, 1] = -1;
            this.RewardMatrix[1, 2] = -1;
            //Console.WriteLine("diamante da pos 4 parou de existir");
            //Console.WriteLine("diamantes azuis pararam de existir");
        }

        if (this.RewardMatrix[2,2] == -2 && next_state[0, 0] == 11)
        {
            
            //Console.WriteLine("diamante da pos 11 parou de existir");
            this.RewardMatrix[2, 2] = -1;
        }

        if (this.RewardMatrix[3,2] == -2 && next_state[0, 0] == 15)
        {
            this.RewardMatrix[3, 2] = -1;
            //Console.WriteLine("diamante da pos 15 parou de existir");
        }

        return (next_state, reward, done);
    }

}


public class simple_env_lantern
{

    private int[,] _rewardMatrix = { { -1, 10, 2, 5 }, { -1, 3, 5, 5 }, { -1, -1, -2, 5 }, { -1, -1, -2, 5 } };

    public int[,] RewardMatrix
    {
        get { return _rewardMatrix; }
        set { _rewardMatrix = value; }
    }
    public int calc_reward(float[,] next_state)

    {
        int row = ((int)next_state[0, 0]-1)/4;
        int col = ((int)next_state[0, 0]-1)%4;
        return this.RewardMatrix[row, col];

    }

    public (float[,], bool) reset()
    {
        var done = false;
        float[,] state = new float[,] { { 10, -1, -2, 3, -1 } };
        this.RewardMatrix[0, 0] = -1;
        this.RewardMatrix[0, 1] = 10;
        this.RewardMatrix[0, 2] = 2;
        this.RewardMatrix[0, 3] = 5;
        this.RewardMatrix[1, 0] = -1;
        this.RewardMatrix[1, 1] = 3;
        this.RewardMatrix[1, 2] = 5;
        this.RewardMatrix[1, 3] = 5;
        this.RewardMatrix[2, 0] = -1;
        this.RewardMatrix[2, 1] = -1;
        this.RewardMatrix[2, 2] = -2;
        this.RewardMatrix[2, 3] = 5;
        this.RewardMatrix[3, 0] = -1;
        this.RewardMatrix[3, 1] = -1;
        this.RewardMatrix[3, 2] = -2;
        this.RewardMatrix[3, 3] = 5;
        return (state, done);

    }



    public (float[,], int, bool) step(float[,] state, int action)
    {
        int reward = 0;
        float[,] next_state = new float[,] { { state[0, 0], state[0, 1], state[0, 2], state[0,3], state[0,4] } };
        var done = false;


        if (action == 0) // esquerda
        {
            if (state[0, 0] == 1)
            {
                next_state[0, 0] = 4;
            }

            else if (state[0, 0] == 3)
            {

                next_state[0, 0] = 2;
                done = true;
            }

            else
            {
                next_state[0, 0] -= 1;
            }

            reward = calc_reward(next_state);

        }



        else if (action == 1) // direita 
        {
            //Console.WriteLine("andei pra direita");
            if (state[0, 0] == 16)
            {
                next_state[0, 0] = 13;
            }

            else if (state[0, 0] == 1)
            {
                next_state[0, 0] = 2;
                done = true;
            }

            else
            {
                next_state[0, 0] += 1;
            }
            reward = calc_reward(next_state);

        }

        else if (action == 2) // cima 
        {
            //Console.WriteLine("andei pra cima");
            if (state[0, 0] == 1)
            {
                next_state[0, 0] = 13;
            }

            else if (state[0, 0] == 3)
            {
                next_state[0, 0] = 15;
            }

            else if (state[0, 0] == 4)
            {
                next_state[0, 0] = 16;
            }

            else if (state[0, 0] == 6)
            {
                next_state[0, 0] = 2;
                done = true;
            }

            else
            {
                next_state[0, 0] -= 4;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 3) // baixo
        {
            //Console.WriteLine("andei pra baixo");
            if (state[0, 0] == 13)
            {
                next_state[0, 0] = 1;
            }

            else if (state[0, 0] == 14)
            {
                next_state[0, 0] = 2;
                done=true;
            }

            else if (state[0, 0] == 15)
            {
                next_state[0, 0] = 3;
            }

            else if (state[0, 0] == 16)
            {
                next_state[0, 0] = 4;
            }

            else
            {
                next_state[0, 0] += 4;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 4) // diagonal cima direita
        {
            //Console.WriteLine("andei pra diagonal cima direita");
            if (state[0, 0] == 1)
            {
                next_state[0, 0] = 14;
            }

            else if (state[0, 0] == 3)
            {
                next_state[0, 0] = 16;
            }


            else if (state[0, 0] == 5)
            {
                next_state[0, 0] = 2;
                done = true;
            }

            else
            {
                next_state[0, 0] -= 3;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 5) // diagonal cima esquerda
        {
            if (state[0, 0] == 5)
            {
                next_state[0, 0] = 4;
            }

            else if (state[0, 0] == 4)
            {
                next_state[0, 0] = 15;
            }

            else if (state[0, 0] == 3)
            {
                next_state[0, 0] = 14;
            }

            else if (state[0, 0] == 1)
            {
                next_state[0, 0] = 16;
            }

            else if (state[0, 0] == 7)
            {
                next_state[0, 0] = 2;
                done=true;
            }

            else
            {
                next_state[0, 0] -= 5;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 6) // diagonal baixo esquerda
        {

            if (state[0, 0] == 14)
            {
                next_state[0, 0] = 1;
            }

            else if (state[0, 0] == 15)
            {
                next_state[0, 0] = 2;
                done = true;
            }

            else if (state[0, 0] == 16)
            {
                next_state[0, 0] = 3;
            }

            else
            {
                next_state[0, 0] += 3;
            }
            reward = calc_reward(next_state);
        }

        else if (action == 7) // diagonal baixo direita
        {
            if (state[0, 0] == 12)
            {
                next_state[0, 0] = 13;
            }

            else if (state[0, 0] == 13)
            {
                next_state[0, 0] = 2;
                done = true;
            }

            else if (state[0, 0] == 14)
            {
                next_state[0, 0] = 3;
            }

            else if (state[0, 0] == 15)
            {
                next_state[0, 0] = 4;
            }

            else if (state[0, 0] == 16)
            {
                next_state[0, 0] = 1;
            }

            else
            {
                next_state[0, 0] += 5;
            }
            reward = calc_reward(next_state);
        }

        if (next_state[0, 0] == 2 || next_state[0, 0] == 13)
        {
            done = true;
            //Console.WriteLine("caiu no fogo ou ganhou");
        }

        if (next_state[0,0] == 1)
        {
            next_state[0, 1] = this.RewardMatrix[0,3]; // mecanismo NAO normal de ver oq ta na esquerda
            next_state[0, 2] = this.RewardMatrix[((int)next_state[0, 0])/4, ((int)next_state[0, 0])%4]; // mecanismo normal de ver oq ta na direita
            next_state[0, 3] = this.RewardMatrix[((int)next_state[0, 0]+11)/4, ((int)next_state[0, 0]+11)%4]; // mecanismo NAO normal de ver oq ta em cima
            next_state[0, 4] = this.RewardMatrix[((int)next_state[0, 0]+3)/4, ((int)next_state[0, 0]+3)%4]; // mecanismo normal de ver oq ta embaixo

        }

        if (next_state[0,0] == 2 || next_state[0,0] == 3 || next_state[0,0] == 4)
        {
            next_state[0, 1] = this.RewardMatrix[((int)next_state[0, 0]-2)/4, ((int)next_state[0, 0]-2)%4]; // mecanismo normal de ver oq ta na esquerda
            next_state[0, 2] = this.RewardMatrix[((int)next_state[0, 0])/4, ((int)next_state[0, 0])%4]; // mecanismo normal de ver oq ta na direita
            next_state[0, 3] = this.RewardMatrix[((int)next_state[0, 0]+11)/4, ((int)next_state[0, 0]+11)%4]; // mecanismo NAO normal de ver oq ta em cima
            next_state[0, 4] = this.RewardMatrix[((int)next_state[0, 0]+3)/4, ((int)next_state[0, 0]+3)%4]; // mecanismo normal de ver oq ta embaixo
        }

        if (next_state[0,0] == 5 || next_state[0, 0] == 6 || next_state[0, 0] == 7 || next_state[0, 0] == 8 || next_state[0, 0] == 9 || next_state[0, 0] == 10 || next_state[0, 0] == 11 || next_state[0, 0] == 12)
        {
            next_state[0, 1] = this.RewardMatrix[((int)next_state[0, 0]-2)/4, ((int)next_state[0, 0]-2)%4]; // mecanismo normal de ver oq ta na esquerda
            next_state[0, 2] = this.RewardMatrix[((int)next_state[0, 0])/4, ((int)next_state[0, 0])%4]; // mecanismo normal de ver oq ta na direita
            next_state[0, 3] = this.RewardMatrix[((int)next_state[0, 0]-5)/4, ((int)next_state[0, 0]-5)%4]; // mecanismo normal de ver oq ta em cima
            next_state[0, 4] = this.RewardMatrix[((int)next_state[0, 0]+3)/4, ((int)next_state[0, 0]+3)%4]; // mecanismo normal de ver oq ta embaixo
        }

        if (next_state[0, 0] == 13 || next_state[0, 0] == 14 || next_state[0, 0] == 15)
        {
            next_state[0, 1] = this.RewardMatrix[((int)next_state[0, 0]-2)/4, ((int)next_state[0, 0]-2)%4]; // mecanismo normal de ver oq ta na esquerda
            next_state[0, 2] = this.RewardMatrix[((int)next_state[0, 0])/4, ((int)next_state[0, 0])%4]; // mecanismo normal de ver oq ta na direita
            next_state[0, 3] = this.RewardMatrix[((int)next_state[0, 0]-5)/4, ((int)next_state[0, 0]-5)%4]; // mecanismo normal de ver oq ta em cima
            next_state[0, 4] = this.RewardMatrix[((int)next_state[0, 0]-13)/4, ((int)next_state[0, 0]-13)%4]; // mecanismo NAO normal de ver oq ta embaixo
        }

        if (next_state[0,0] == 16)
        {
            next_state[0, 1] = this.RewardMatrix[((int)next_state[0, 0]-2)/4, ((int)next_state[0, 0]-2)%4]; // mecanismo normal de ver oq ta na esquerda
            next_state[0, 2] = this.RewardMatrix[3,0]; // mecanismo NAO normal de ver oq ta na direita
            next_state[0, 3] = this.RewardMatrix[((int)next_state[0, 0]-5)/4, ((int)next_state[0, 0]-5)%4]; // mecanismo normal de ver oq ta em cima
            next_state[0, 4] = this.RewardMatrix[((int)next_state[0, 0]-13)/4, ((int)next_state[0, 0]-13)%4]; // mecanismo NAO normal de ver oq ta embaixo
        }

        if (this.RewardMatrix[0, 2]==2 && next_state[0, 0] == 3)
        {
            this.RewardMatrix[0, 2] = -1;
            this.RewardMatrix[0, 3] = -1;
            this.RewardMatrix[1, 3] = -1;
            this.RewardMatrix[2, 3] = -1;
            this.RewardMatrix[3, 3] = -1;
            //Console.WriteLine("diamante da pos 3 parou de existir");
            //Console.WriteLine("diamantes amarelos pararam de existir");
        }

        if (this.RewardMatrix[1, 1] == 3 && next_state[0, 0] == 6)
        {
            this.RewardMatrix[1, 1] = -1;
            this.RewardMatrix[0, 3] = -1;
            this.RewardMatrix[1, 3] = -1;
            this.RewardMatrix[2, 3] = -1;
            this.RewardMatrix[3, 3] = -1;
            //Console.WriteLine("diamante da pos 6 parou de existir");
            //Console.WriteLine("diamantes amarelos pararam de existir");
        }

        if (this.RewardMatrix[1, 2] == 5 && next_state[0, 0] == 7)
        {
            this.RewardMatrix[1, 2] = -1;
            this.RewardMatrix[0, 3] = -1;
            this.RewardMatrix[1, 3] = -1;
            this.RewardMatrix[2, 3] = -1;
            this.RewardMatrix[3, 3] = -1;
            //Console.WriteLine("diamante da pos 7 parou de existir");
            //Console.WriteLine("diamantes amarelos pararam de existir");
        }

        if (this.RewardMatrix[3, 3]==5 && next_state[0, 0] == 16)
        {
            this.RewardMatrix[3, 3] = -1;
            this.RewardMatrix[0, 2] = -1;
            this.RewardMatrix[1, 1] = -1;
            this.RewardMatrix[1, 2] = -1;
            //Console.WriteLine("diamante da pos 16 parou de existir");
            //Console.WriteLine("diamantes azuis pararam de existir");
        }

        if (this.RewardMatrix[2, 3] == 5 && next_state[0, 0] == 12)
        {
            this.RewardMatrix[2, 3] = -1;
            this.RewardMatrix[0, 2] = -1;
            this.RewardMatrix[1, 1] = -1;
            this.RewardMatrix[1, 2] = -1;
            //Console.WriteLine("diamante da pos 12 parou de existir");
            //Console.WriteLine("diamantes azuis pararam de existir");
        }

        if (this.RewardMatrix[1, 3]==5 && next_state[0, 0] == 8)
        {
            this.RewardMatrix[1, 3] = -1;
            this.RewardMatrix[0, 2] = -1;
            this.RewardMatrix[1, 1] = -1;
            this.RewardMatrix[1, 2] = -1;
            //Console.WriteLine("diamante da pos 8 parou de existir");
            //Console.WriteLine("diamantes azuis pararam de existir");
        }

        if (this.RewardMatrix[0, 3] == 5 && next_state[0, 0] == 4)
        {
            this.RewardMatrix[0, 3] = -1;
            this.RewardMatrix[0, 2] = -1;
            this.RewardMatrix[1, 1] = -1;
            this.RewardMatrix[1, 2] = -1;
            //Console.WriteLine("diamante da pos 4 parou de existir");
            //Console.WriteLine("diamantes azuis pararam de existir");
        }

        if (this.RewardMatrix[2, 2] == -2 && next_state[0, 0] == 11)
        {

            //Console.WriteLine("diamante da pos 11 parou de existir");
            this.RewardMatrix[2, 2] = -1;
        }

        if (this.RewardMatrix[3, 2] == -2 && next_state[0, 0] == 15)
        {
            this.RewardMatrix[3, 2] = -1;
            //Console.WriteLine("diamante da pos 15 parou de existir");
        }

        return (next_state, reward, done);
    }

}


public class replay_memory
{
  public List<(float[,] state, int action, float reward, float[,] next_state, bool done)> memory;
  public Random rnd;
  public replay_memory(int capacity = 2000)
  {
    // For now, capacity is not working... Improve this later.
    memory = new();
    rnd = new();
  }
  public void remember(float[,] state, int action, float reward, float[,] next_state, bool done)
  {
    this.memory.Add((state, action, reward, next_state, done));
  }



  public List<(float[,], int, float, float[,], bool)> sample(int batch_size)
  {
    List<(float[,] state, int action, float reward, float[,] next_state, bool done)> selected = this.memory.Select(x => x).OrderBy(x => rnd.Next()).Take(batch_size).ToList();
    return selected;
  }

  public int size
  {
    get
    {
      return this.memory.Count;
    }
  }

}





