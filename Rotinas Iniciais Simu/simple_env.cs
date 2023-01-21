using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Rotinas_Iniciais_Simu
{
  public class simple_env
  {
    public int[] state;
    public int calc_reward(float state, float next_state)
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

      if (next_state == 1 ^ next_state == 5 ^ next_state == 9 ^ next_state == 13 ^ next_state == 14 ^ next_state == 10)
      {
        return 0;
      }

      else if (next_state == 11 & this.state[8 - 1] == 1)
      {
        return -2;
      }

      else if (next_state == 11 & this.state[8 - 1] == 0)
      {
        return 0;
      }

      else if (next_state == 15 & this.state[9 - 1] == 1)
      {
        return -2;
      }

      else if (next_state == 15 & this.state[9 - 1] == 0)
      {
        return 0;
      }

      else if (next_state == 16 & this.state[4 - 1] == 1)
      {
        return 4;
      }

      else if (next_state == 16 & this.state[4 - 1] == 0)
      {
        return 0;
      }

      else if (next_state == 12 & this.state[5 - 1] == 1)
      {
        return 4;
      }

      else if (next_state == 12 & this.state[5 - 1] == 0)
      {
        return 0;
      }

      else if (next_state == 8 & this.state[6 - 1] == 1)
      {
        return 4;
      }

      else if (next_state == 8 & this.state[6 - 1] == 0)
      {
        return 0;
      }

      else if (next_state == 4 & this.state[7 - 1] == 1)
      {
        return 4;
      }

      else if (next_state == 4 & this.state[7 - 1] == 0)
      {
        return 0;
      }

      else if (next_state == 6 & this.state[2 - 1] == 1)
      {
        return 3;
      }

      else if (next_state == 6 & this.state[2 - 1] == 0)
      {
        return 0;
      }

      else if (next_state == 7 & this.state[3 - 1] == 1)
      {
        return 5;
      }

      else if (next_state == 7 & this.state[3 - 1] == 0)
      {
        return 0;
      }

      else if (next_state == 3 & this.state[1 - 1] == 1)
      {
        return 2;
      }

      else if (next_state == 3 & this.state[1 - 1] == 0)
      {
        return 0;
      }

      else if (next_state == 2)
      {
        return 10;
      }

      else
      {
        return 0;
      }
    }

    public (int, bool) reset()
    {
      var done = false;
      int state = 10;
      this.state = new int[]{1,1,1,1,1,1,1,1,1};

      return (state, done);

    }



    public (int, int, bool) step(int state, int action)
    {
      var next_state = state;
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
          switch (state)
          {
            case 1 or 5 or 9 or 13:
              next_state += 3;
              break;
            default:
              next_state -= 1;
              break;
          }
          break;


        case 1: // direita
          switch (state)
          {
            case 4 or 8 or 12 or 16:
              next_state -= 3;
              break;
            default:
              next_state += 1;
              break;
          }
          break;

        case 2: // cima
          switch (state)
          {
            case 1 or 2 or 3 or 4:
              next_state += 12;
              break;
            default:
              next_state -= 4;
              break;
          }
          break;
        case 3: // baixo
          switch (state)
          {
            case 13 or 14 or 15 or 16:
              next_state -= 12;
              break;
            default:
              next_state += 4;
              break;
          }
          break;

        default:
          break;
      }

      if (next_state <= 0 | next_state >= 17)
        throw new Exception("next state out of possible range");

      if (next_state == 2 || next_state == 13)
      {
        done = true;
        //Console.WriteLine("caiu no fogo ou ganhou");
      }
      if (this.state[1 - 1] == 1 && next_state == 3)
      {
        this.state[1 - 1] = 0;
        this.state[4 - 1] = 0;
        this.state[5 - 1] = 0;
        this.state[6 - 1] = 0;
        this.state[7 - 1] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (this.state[2 - 1] == 1 && next_state == 6)
      {
        this.state[2 - 1] = 0;
        this.state[4 - 1] = 0;
        this.state[5 - 1] = 0;
        this.state[6 - 1] = 0;
        this.state[7 - 1] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (this.state[3 - 1] == 1 && next_state == 7)
      {
        this.state[3 - 1] = 0;
        this.state[4 - 1] = 0;
        this.state[5 - 1] = 0;
        this.state[6 - 1] = 0;
        this.state[7 - 1] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (this.state[4 - 1] == 1 && next_state == 16)
      {
        this.state[4 - 1] = 0;
        this.state[1 - 1] = 0;
        this.state[2 - 1] = 0;
        this.state[3 - 1] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (this.state[5 - 1] == 1 && next_state == 12)
      {
        this.state[5 - 1] = 0;
        this.state[1 - 1] = 0;
        this.state[2 - 1] = 0;
        this.state[3 - 1] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (this.state[6 - 1] == 1 && next_state == 8)
      {
        this.state[6 - 1] = 0;
        this.state[1 - 1] = 0;
        this.state[2 - 1] = 0;
        this.state[3 - 1] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (this.state[7 - 1] == 1 && next_state == 4)
      {
        this.state[7 - 1] = 0;
        this.state[1 - 1] = 0;
        this.state[2 - 1] = 0;
        this.state[3 - 1] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (this.state[8 - 1] == 1 && next_state == 11)
      {
        this.state[8 - 1] = 0;
        //Console.WriteLine("diamante parou de existir");
      }

      if (this.state[9-1] == 1 && next_state == 15)
      {
        this.state[9 - 1] = 0;
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
