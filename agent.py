import torch as T
from networks import ActorNetwork, CriticNetwork
import numpy as np

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.0001, beta=0.0002, fc1=64,
                    fc2=64, gamma=0.99, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        
        # print(f"Initializing agent {agent_idx} with actor_dims: {actor_dims}")

        # 根据agent_idx选择不同的网络结构
        if agent_idx < 2:  # UAV agents
            self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                    name=f'uav_{agent_idx}_actor', chkpt_dir=chkpt_dir)
        else:  # USV agents
            self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                    name=f'usv_{agent_idx-2}_actor', chkpt_dir=chkpt_dir)
        
        # Critic网络对所有智能体使用相同的结构
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, 
                                name=f'agent_{agent_idx}_critic', chkpt_dir=chkpt_dir)
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                name=f'agent_{agent_idx}_target_actor', chkpt_dir=chkpt_dir)
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions,
                                name=f'agent_{agent_idx}_target_critic', chkpt_dir=chkpt_dir)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, time_step, evaluate=False):
        # print(f"Agent {self.agent_name} received observation of shape: {len(observation)}")
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)

        # exploration
        max_noise = 0.75
        min_noise = 0.01
        decay_rate = 0.999995

        noise_scale = max(min_noise, max_noise * (decay_rate ** time_step))
        noise = 2 * T.rand(self.n_actions).to(self.actor.device) - 1 # [-1,1)
        if not evaluate:
            noise = noise_scale * noise
        else:
            noise = 0 * noise
        
        action = actions + noise
        action_np = action.detach().cpu().numpy()[0]
        magnitude = np.linalg.norm(action_np)
        if magnitude > 0.04:
            action_np = action_np / magnitude * 0.04
        return action_np

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
