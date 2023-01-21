using System;
using System.Collections;
using System.Linq.Expressions;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;

namespace Rotinas_Iniciais_Simu
{
  public class q_learning
  {
    Random rnd;
    float gamma;
    float exploration_max;
    float exploration_min;
    float exploration_decay;
    float learning_rate_Q;
    int n_outputs;
    int n_inputs;
    int episodes;
    int max_episodes;
    float[,,,,,,,,,,] qTable;

    public q_learning()
    {
      Random rnd = new();
      int memory_size = Convert.ToInt32(5 * Math.Pow(10, 5));
      gamma = 1f;
      exploration_max = 1f;
      exploration_min = 0.0f;
      exploration_decay = 0.999f;
      learning_rate_Q = 0.5f;
      n_outputs = 4;
      n_inputs = 10;
      max_episodes = (int)1e5;
      qTable = makeQtable();
      loop(max_episodes, rnd);
    }


    int get_actionQ(float[,] state, Random rnd, float[,,,,,,,,,,] qTable, bool should_explore = true, bool sould_print = false)
    {
      if (should_explore)
      {
        exploration_max *= exploration_decay;
        exploration_max = Math.Max(exploration_min, exploration_max);
        if (torch.rand(1).item<float>() < exploration_max)
          return rnd.Next(3);
      }
      int best_action = argMaxTable(state, qTable);

      return best_action;
    }

    float[,,,,,,,,,,] makeQtable()
    {
      return new float[16, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4];
    }
    
    int argMaxTable(float[,] state, float[,,,,,,,,,,] qTable)
    {
      float max = float.MinValue;
      float compare;
      int action = -1;
      for (int a = 0; a < n_outputs; a++)
      {
        if (state[0, 0] <= 0)
          throw new Exception(" state or next_state with position value incorrect.");

        compare = qTable[(int)state[0, 0] - 1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], a];
        if (compare > max)
        {
          max = compare;
          action = a;
        }
      }
      if (action == -1)
        throw new Exception("Action with invalid value/was not assigned");
      return action;
    }

    float MaxTable(float[,] state, float[,,,,,,,,,,] qTable)
    {
      float max = float.MinValue;
      float compare;
      int action = -1;
      for (int a = 0; a < n_outputs; a++)
      {
        if (state[0, 0] <= 0)
          throw new Exception(" state or next_state with position value incorrect.");

        compare = qTable[(int)state[0, 0] - 1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], a];
        if (compare > max)
        {
          max = compare;
          action = a;
        }
      }
      return max;
    }

    void print_environment(float[,] state, float[,] ns, simple_env env, float reward, float ep_reward, int episode, float old_state_Q, float new_state_Q)
    {
      Console.Clear();
      int pos = (int)state[0, 0];
      Console.WriteLine("Rewards: ");
      for (int i = 1; i <= 16; i++)
      {
        float[,] next_state = (float[,])state.Clone();
        next_state[0, 0] = i;

        if (i == pos)
          Console.BackgroundColor = ConsoleColor.DarkBlue;
        if (ns[0, 0] == i)
          Console.BackgroundColor = ConsoleColor.DarkMagenta;
        if (i == 2)
          Console.BackgroundColor = ConsoleColor.Green;
        if (i == 13)
          Console.BackgroundColor = ConsoleColor.Red;
        Console.Write(String.Format("{0, 5}", env.calc_reward(state, next_state)));
        Console.BackgroundColor = ConsoleColor.Black;
        if (i%4 == 0)
          Console.WriteLine();
      }
      
      Console.WriteLine($"Episode: {episode}");
      Console.WriteLine($"Ep. Reward: {ep_reward}");
      Console.WriteLine($"Reward: {reward}");
      Console.WriteLine($"Exploration Rate: {exploration_max}");
      string fmt = "{0,9:0.##}";
      Console.Write("Old Q: ");
      Console.WriteLine(fmt, old_state_Q);
      Console.Write("New Q: ");
      Console.WriteLine(fmt, new_state_Q);
      Console.Write("Diff : ");
      Console.WriteLine(fmt, old_state_Q - new_state_Q);


      float max = float.MinValue;
      foreach (var i in qTable)
      {
        if (i >= max)
          max = i;
      }
      Console.WriteLine($"Max value of Q-Table: {max}");


      //// Print Q table

      //Console.WriteLine("Q-Table: ");
      //var s = (float[,])state.Clone();
      //for (int i = 1; i <= 16; i++)
      //{
      //  s[0, 0] = i;
      //  Console.Write(String.Format("{0, 5}", argMaxTable(s, qTable)));
      //  if (i % 4 == 0)
      //    Console.WriteLine();
      //}

      // Print state action values
      Console.WriteLine("State action values: L R U D");
      for (int a = 0; a < 4; a++)
        Console.Write(String.Format("{0, 5:0.00}", qTable[(int)state[0, 0] - 1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], a]));

      System.Threading.Thread.Sleep(50);
    }

    void loop(int max_episodes, Random rnd)
    {
      Console.WriteLine("Começando Q-Learning...");
      var env = new simple_env();
      var (state, done) = env.reset();
      float[,] next_state;
      float reward;
      for (int ep = 0; ep < max_episodes; ep++)
      {
        float ep_reward = 0;
        (state, done) = env.reset();
        int iter = 0;
        int action;
        while (!done)
        {
          iter++;
          action = get_actionQ(state, rnd, qTable);

          (next_state, reward, done) = env.step(state, action);
          

          float old_state_Q = qTable[(int)state[0, 0] - 1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], action];
          if (state[0, 0] <= 0 | next_state[0, 0] <= 0)
            throw new Exception(" state or next_state with position value incorrect.");
          float new_state_Q = (reward + gamma * MaxTable(next_state, qTable));
          qTable[
            (int)state[0, 0] - 1,
            (int)state[0, 1],
            (int)state[0, 2],
            (int)state[0, 3],
            (int)state[0, 4],
            (int)state[0, 5],
            (int)state[0, 6],
            (int)state[0, 7],
            (int)state[0, 8],
            (int)state[0, 9],
            action] = (1-learning_rate_Q) * old_state_Q  + learning_rate_Q * new_state_Q;
          
          print_environment(state, next_state, env, reward, ep_reward, ep, old_state_Q, new_state_Q);
          ep_reward += reward;
          state = (float[,])next_state.Clone();
          
        }
      }
    }
  }
}
