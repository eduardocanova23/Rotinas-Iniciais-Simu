﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;

namespace Rotinas_Iniciais_Simu
{
  public class QLearning
  {
    Random rnd;
    float gamma = 1.0f;
    float exploration_max = 1.0f;
    float exploration_min = 0.01f;
    float exploration_decay = 0.9999f;
    float learning_rate_Q = 0.3f;
    int n_outputs = 4;
    int n_inputs = 10;
    int episodes = (int)1e5;

    public QLearning()
    {
      Random rnd = new();
      int memory_size = Convert.ToInt32(5 * Math.Pow(10, 5));
      int batch_size = 32;
      float gamma = 1.0f;
      float exploration_max = 1.0f;
      float exploration_min = 0.01f;
      float exploration_decay = 0.9999f;
      float learning_rate = 0.003f;
      float learning_rate_Q = 0.3f;
      float tau = 0.125f;
      int n_outputs = 4;
      int n_inputs = 10;
      int layers = 2;
      int neurons = 32;
      int max_episodes = (int)1e5;

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
          continue;
        compare = qTable[(int)state[0, 0] - 1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], a];
        if (compare > max)
        {
          max = compare;
          action = a;
        }
      }
      return action;
    }

    void loop(int max_episodes, Random rnd)
    {
      Console.WriteLine("Começando Q-Learning...");
      var env = new simple_env();
      var (state, done) = env.reset();
      float[,] next_state;
      float reward;
      var qTable = makeQtable();
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
          ep_reward += reward;
          float state_value = qTable[(int)state[0, 0] - 1, (int)state[0, 1], (int)state[0, 2], (int)state[0, 3], (int)state[0, 4], (int)state[0, 5], (int)state[0, 6], (int)state[0, 7], (int)state[0, 8], (int)state[0, 9], action];
          if (state[0,0] <= 0 | next_state[0,0] <= 0)
            continue;

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
            action] = learning_rate_Q * (reward + gamma * argMaxTable(next_state, qTable) - state_value);
          state = (float[,])next_state.Clone();
        }
        Console.WriteLine($"Episode reward during training: {ep_reward}");
      }
    }
  }
}
