import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from env.mdp import StochasticMDPEnv
from agent.Hdqn import Hdqn
from utils.plotting import plot_episode_stats, plot_visited_states
from utils import plotting
plt.style.use('ggplot')

def one_hot(state):
    vector = np.zeros(6)
    vector[state-1] = 1.0
    return np.expand_dims(vector, axis=0)

def main():
    ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
    MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state", "done"])
    env = StochasticMDPEnv()
    agent = Hdqn()
    visits = np.zeros((12, 6))
    goals = np.zeros((12, 6))
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(12000),
        episode_rewards=np.zeros(12000))
    
    anneal_factor = (1.0-0.1)/12000 
    print "Annealing factor: " + str(anneal_factor)
    for episode_thousand in range(12):
        for episode in range(1000):
            episode_length = 0
            print "\n\n### EPISODE "  + str(episode_thousand*1000 + episode) + "###"
            state = env.reset()
            visits[episode_thousand][state-1] += 1
            done = False
            while not done:
                goal = agent.select_goal(one_hot(state))[0]
                agent.goal_selected[goal] += 1
                goals[episode_thousand][goal]+=1
                print "\nNew Goal: "  + str(goal+1) + "\nState-Actions: "
                total_external_reward = 0
                goal_reached = False
                while not done and not goal_reached:
                    episode_length += 1
                    action = agent.select_move(one_hot(state), one_hot(goal+1), goal)[0]
                    print(str((state,action)) + "; ")
                    next_state, external_reward, done = env.step(action)
                    if external_reward==1:
                        print "extrinsic_reward: ", goal+1," reward:", external_reward
                    #print "next_state, external_reward, done", next_state, external_reward, done
                    # Update statistics
                    stats.episode_rewards[episode_thousand*1000 + episode] += external_reward
                    stats.episode_lengths[episode_thousand*1000 + episode] = episode_length

                    visits[episode_thousand][next_state-1] += 1
                    intrinsic_reward = agent.criticize(goal+1, next_state)
                    goal_reached = next_state == goal+1
                    if goal_reached:
                        agent.goal_success[goal] += 1
                        print "Goal reached!! "
                    if next_state == 6:
                        print "S6 reached!! "
                    exp = ActorExperience(one_hot(state), one_hot(goal+1), action, intrinsic_reward, one_hot(next_state), done)
                    agent.store(exp, meta=False)
                    agent.update(meta=False)
                    agent.update(meta=True)
                    total_external_reward += external_reward
                    state = next_state
                exp = MetaExperience(one_hot(state), goal, total_external_reward, one_hot(next_state), done)
                agent.store(exp, meta=True)
                
                #Annealing 
                agent.meta_epsilon -= anneal_factor
                avg_success_rate = agent.goal_success[goal] / agent.goal_selected[goal]
                print "avg_success_rate : ", avg_success_rate
                # if(avg_success_rate < 0.9):
                agent.actor_epsilon[goal] -= anneal_factor
                # else:
                #     agent.actor_epsilon[goal] = 1 - avg_success_rate
            
                if agent.actor_epsilon[goal] < 0.1:
                    agent.actor_epsilon[goal] = 0.1
                if agent.meta_epsilon < 0.1:
                    agent.meta_epsilon = 0.1
                print "meta_epsilon: " + str(agent.meta_epsilon)
                print "actor_epsilon " + str(goal + 1) + ": " + str(agent.actor_epsilon[goal])

                
        print "visits", visits
    
    print "goals", goals
    fig1,fig2,fig3 = plot_episode_stats(stats)

    plot_visited_states(visits, 12000)

    eps = list(range(1,13))
    plt.subplot(2, 3, 1)
    plt.plot(eps, visits[:,0]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S1")
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(eps, visits[:,1]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S2")
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(eps, visits[:,2]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(0.0, 1.0)
    plt.xlim(1, 12)
    plt.title("S3")
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(eps, visits[:,3]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(0.0, 1.0)
    plt.xlim(1, 12)
    plt.title("S4")
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(eps, visits[:,4]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(0, 1.0)
    plt.xlim(1, 12)
    plt.title("S5")
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(eps, visits[:,5]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(0, 1.0)
    plt.xlim(1, 12)
    plt.title("S6")
    plt.grid(True)
    plt.savefig('first_run.png')
    plt.show()  

    plt.clf()
    
    eps = list(range(1,13))
    plt.subplot(2, 3, 1)
    plt.plot(eps, goals[:,0]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S1")
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(eps, goals[:,1]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S2")
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(eps, goals[:,2]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(0.0, 1.0)
    plt.xlim(1, 12)
    plt.title("S3")
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(eps, goals[:,3]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(0.0, 1.0)
    plt.xlim(1, 12)
    plt.title("S4")
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(eps, goals[:,4]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(0, 1.0)
    plt.xlim(1, 12)
    plt.title("S5")
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(eps, goals[:,5]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(0, 1.0)
    plt.xlim(1, 12)
    plt.title("S6")
    plt.grid(True)
    plt.savefig('first_run_goals.png')
    plt.show()
if __name__ == "__main__":
    main()
