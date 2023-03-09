using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Rotinas_Iniciais_Simu
{
    public class env_incomplete_two_diamonds
    {

        private int[,] _rewardMatrix = { { -1, 10, -1, -1 }, { -1, 3, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, 11 } };

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

        public (float[,], bool) reset(bool train = true)
        {
            Random rnd = new Random();
            int rand_pos = 2;
            var done = false;
            float[,] state;
            if (train)
            {
                state = new float[,] { { 10 } };
                while (rand_pos ==2 || rand_pos ==13)
                {
                    rand_pos = rnd.Next(1, 17);
                    state = new float[,] { { rand_pos } };

                }
                this.RewardMatrix[0, 0] = -1;
                this.RewardMatrix[0, 1] = 10;
                this.RewardMatrix[0, 2] = -1;
                this.RewardMatrix[0, 3] = -1;
                this.RewardMatrix[1, 0] = -1;
                this.RewardMatrix[1, 1] = 3;
                this.RewardMatrix[1, 2] = -1;
                this.RewardMatrix[1, 3] = -1;
                this.RewardMatrix[2, 0] = -1;
                this.RewardMatrix[2, 1] = -1;
                this.RewardMatrix[2, 2] = -1;
                this.RewardMatrix[2, 3] = -1;
                this.RewardMatrix[3, 0] = -1;
                this.RewardMatrix[3, 1] = -1;
                this.RewardMatrix[3, 2] = -1;
                this.RewardMatrix[3, 3] = 11;
                return (state, done);
            }

            else
            {
                state = new float[,] { { 10 } };
                this.RewardMatrix[0, 0] = -1;
                this.RewardMatrix[0, 1] = 10;
                this.RewardMatrix[0, 2] = -1;
                this.RewardMatrix[0, 3] = -1;
                this.RewardMatrix[1, 0] = -1;
                this.RewardMatrix[1, 1] = 3;
                this.RewardMatrix[1, 2] = -1;
                this.RewardMatrix[1, 3] = -1;
                this.RewardMatrix[2, 0] = -1;
                this.RewardMatrix[2, 1] = -1;
                this.RewardMatrix[2, 2] = -1;
                this.RewardMatrix[2, 3] = -1;
                this.RewardMatrix[3, 0] = -1;
                this.RewardMatrix[3, 1] = -1;
                this.RewardMatrix[3, 2] = -1;
                this.RewardMatrix[3, 3] = 11;
                return (state, done);
            }

        }



        public (float[,], int, bool) step(float[,] state, int action)
        {
            int reward = 0;
            float[,] next_state = new float[,] { { state[0, 0] } };
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





            if (this.RewardMatrix[1, 1] == 3 && next_state[0, 0] == 6)
            {
                this.RewardMatrix[1, 1] = -1;
                this.RewardMatrix[3, 3] = -1;
                //Console.WriteLine("diamante da pos 6 parou de existir");
                //Console.WriteLine("diamantes amarelos pararam de existir");
            }



            if (this.RewardMatrix[3, 3]==11 && next_state[0, 0] == 16)
            {
                this.RewardMatrix[3, 3] = -1;
                this.RewardMatrix[1, 1] = -1;
                //Console.WriteLine("diamante da pos 16 parou de existir");
                //Console.WriteLine("diamantes azuis pararam de existir");
            }



            return (next_state, reward, done);
        }



    }
}
