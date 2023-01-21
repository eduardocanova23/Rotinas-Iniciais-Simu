using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Rotinas_Iniciais_Simu
{
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

    public int argMax(float[,] state)
    {
      return 1;
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
}
