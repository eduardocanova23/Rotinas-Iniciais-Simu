using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Rotinas_Iniciais_Simu
{
  public class simple_env
  {
    public int calc_reward(float[,] state, float[,] next_state)
    {
      //var _state = new List<float>();
      //var _next_state = new List<float>();
      //for (long i = 0, imax = state.size()[0] + 1; i < imax; i++)
      //{
      //  _state.Add(state[0][i].item<float>());
      //}
      //for (long i = 0, imax = next_state.size()[0]; i < imax; i++)
      //{
      //  _next_state.Add(next_state[0][i].item<float>());
      //}

      if (next_state[0, 0] == 1 ^ next_state[0, 0] == 5 ^ next_state[0, 0] == 9 ^ next_state[0, 0] == 13 ^ next_state[0, 0] == 14 ^ next_state[0, 0] == 10)
      {
        return -1;
      }

      else if (next_state[0, 0] == 11 & state[0, 8] == 1)
      {
        return -2;
      }

      else if (next_state[0, 0] == 11 & state[0, 8] == 0)
      {
        return 0;
      }

      else if (next_state[0, 0] == 15 & state[0, 9] == 1)
      {
        return -2;
      }

      else if (next_state[0, 0] == 15 & state[0, 9] == 0)
      {
        return 0;
      }

      else if (next_state[0, 0] == 16 & state[0, 4] == 1)
      {
        return 4;
      }

      else if (next_state[0, 0] == 16 & state[0, 4] == 0)
      {
        return 0;
      }

      else if (next_state[0, 0] == 12 & state[0, 5] == 1)
      {
        return 4;
      }

      else if (next_state[0, 0] == 12 & state[0, 5] == 0)
      {
        return 0;
      }

      else if (next_state[0, 0] == 8 & state[0, 6] == 1)
      {
        return 4;
      }

      else if (next_state[0, 0] == 8 & state[0, 6] == 0)
      {
        return 0;
      }

      else if (next_state[0, 0] == 4 & state[0, 7] == 1)
      {
        return 4;
      }

      else if (next_state[0, 0] == 4 & state[0, 7] == 0)
      {
        return 0;
      }

      else if (next_state[0, 0] == 6 & state[0, 2] == 1)
      {
        return 3;
      }

      else if (next_state[0, 0] == 6 & state[0, 2] == 0)
      {
        return 0;
      }

      else if (next_state[0, 0] == 7 & state[0, 3] == 1)
      {
        return 5;
      }

      else if (next_state[0, 0] == 7 & state[0, 3] == 0)
      {
        return 0;
      }

      else if (next_state[0, 0] == 3 & state[0, 1] == 1)
      {
        return 2;
      }

      else if (next_state[0, 0] == 3 & state[0, 1] == 0)
      {
        return 0;
      }

      else if (next_state[0, 0] == 2)
      {
        return 10;
      }

      else
      {
        return 0;
      }
    }

    public (float[,], bool) reset()
    {
      var done = false;
      float[,] state = new float[,] { { 10, 1, 1, 1, 1, 1, 1, 1, 1, 1 } };

      return (state, done);

    }



    public (float[,], int, bool) step(float[,] state, int action)
    {
      float[,] next_state = new float[,] { { state[0, 0], state[0, 1], state[0, 2], state[0, 3], state[0, 4], state[0, 5], state[0, 6], state[0, 7], state[0, 8], state[0, 9] } };
      //var _state = new List<float>();
      //var _next_state = new List<float>();
      var done = false;

      //for (long i = 0, imax = state.size()[0] + 1; i < imax; i++)
      //{
      //  _state.Add(state[0][i].item<float>());
      //  _next_state.Add((float)state[0][i].item<float>());
      //}

      switch (action)
      {
        case 0: // esquerda
          switch (state[0, 0])
          {
            case 1 or 5 or 9 or 13:
              next_state[0, 0] += 3;
              break;
            default:
              next_state[0, 0] -= 1;
              break;
          }
          break;


        case 1: // direita
          switch (state[0, 0])
          {
            case 4 or 8 or 12 or 16:
              next_state[0, 0] -= 3;
              break;
            default:
              next_state[0, 0] += 1;
              break;
          }
          break;

        case 2: // cima
          switch (state[0, 0])
          {
            case 1 or 2 or 3 or 4:
              next_state[0, 0] += 12;
              break;
            default:
              next_state[0, 0] -= 4;
              break;
          }
          break;
        case 3: // baixo
          switch (state[0, 0])
          {
            case 13 or 14 or 15 or 16:
              next_state[0, 0] -= 12;
              break;
            default:
              next_state[0, 0] += 4;
              break;
          }
          break;

        default:
          break;
      }

      if (next_state[0, 0] <= 0 | next_state[0, 0] >= 17)
        next_state[0, 0] = 1;

      if (next_state[0, 0] == 2 || next_state[0, 0] == 13)
      {
        done = true;
        //Console.WriteLine("caiu no fogo ou ganhou");
      }
      if (next_state[0, 1] == 1 && next_state[0, 0] == 3)
      {
        next_state[0, 1] = 0;
        next_state[0, 4] = 0;
        next_state[0, 5] = 0;
        next_state[0, 6] = 0;
        next_state[0, 7] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (next_state[0, 2] == 1 && next_state[0, 0] == 6)
      {
        next_state[0, 2] = 0;
        next_state[0, 4] = 0;
        next_state[0, 5] = 0;
        next_state[0, 6] = 0;
        next_state[0, 7] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (next_state[0, 3] == 1 && next_state[0, 0] == 7)
      {
        next_state[0, 3] = 0;
        next_state[0, 4] = 0;
        next_state[0, 5] = 0;
        next_state[0, 6] = 0;
        next_state[0, 7] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (next_state[0, 4] == 1 && next_state[0, 0] == 16)
      {
        next_state[0, 4] = 0;
        next_state[0, 1] = 0;
        next_state[0, 2] = 0;
        next_state[0, 3] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (next_state[0, 5] == 1 && next_state[0, 0] == 12)
      {
        next_state[0, 5] = 0;
        next_state[0, 1] = 0;
        next_state[0, 2] = 0;
        next_state[0, 3] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (next_state[0, 6] == 1 && next_state[0, 0] == 8)
      {
        next_state[0, 6] = 0;
        next_state[0, 1] = 0;
        next_state[0, 2] = 0;
        next_state[0, 3] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (next_state[0, 7] == 1 && next_state[0, 0] == 4)
      {
        next_state[0, 7] = 0;
        next_state[0, 1] = 0;
        next_state[0, 2] = 0;
        next_state[0, 3] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (next_state[0, 8] == 1 && next_state[0, 0] == 11)
      {
        next_state[0, 8] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (next_state[0, 9] == 1 && next_state[0, 0] == 15)
      {
        next_state[0, 9] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      int reward = calc_reward(state, next_state);

      //state.print();
      //next_state.print();
      //Console.WriteLine($"action: {action}, reward: {reward}");
      //Console.WriteLine("-=--=--=--=--=--=--=--=-");

      return (next_state, reward, done);
    }

  }
}
