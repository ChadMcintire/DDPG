The paper can be found at https://arxiv.org/pdf/1802.09477.pdf.

Original github at https://github.com/sfujim/TD3

Because the code seems to be base off the spinning up code rather than the original paper, 
we will follow the spinning up pseudocode in our code base. The original code is posted for completeness.
More can be found here:
https://spinningup.openai.com/en/latest/algorithms/ddpg.html

OpenAI Spinning Up Pseudocode
=============================

$\textbf{Algorithm 1}$ Deep Deterministic Policy Gradient
1. Input: Initialize policy $\theta$, Q-function parameters $\phi$, empty replay buff $\mathcal{D}$
2. Set target parameter equal to main parameters $\theta_{targ} \leftarrow \theta, \phi_{targ} \leftarrow \phi$
3. $\textbf{repeat}$
    1. Observe state $s$ and selection action $a$ = clip( $\mu_\theta(s) + \epsilon, a_{Low}, a_{High}$ ), where $\epsilon \sim \mathcal{N}$
    2. Execute $a$ in the environment 
    3. Observe next state $s\prime$, reward $r$, and done signal $d$ to indicate whether $s\prime$ is terminal
    4. Store $(s, a, r, s\prime, d)$ in replay buffer $\mathcal{D}$
    5. If $s\prime$ is terminal, reset environment state.
    6. $\textbf{if}$ it's time to update $\textbf{then}$
         1. $\textbf{for}$ however many updates $\textbf{do}$
             1. Randomly sample a batch of transitions, $B = \{(s,a,r,s\prime,d)\} for \mathcal{D}$
             2. Compute targets<br />
                $y(r, s\prime, d) = r + \gamma(1 - d) Q_{\phi_{targ}}(s\prime, \mu_{\theta_{targ}}(s\prime))$
             3. Update Q-function by one step of gradient descent using <br />
                $\nabla_{\phi}\frac{1}{|B|}\sum_{(s,a,r,s\prime,d) \in B} (Q_\phi(s,a) - y(r,s\prime, d))^2$
             4. Update policy by one step of gradient ascent using<br />
                $\nabla_{theta} \frac{1}{|B|}\sum_{s \in B} Q_{\phi}(s, \mu_\theta(s))$
             5. Update target networks with <br />
                $\phi_{targ} \leftarrow \rho\phi_{targ} + (1 - \rho)\phi$<br />
                $\theta_{targ} \leftarrow \rho \theta_{targ} + (1 - \rho)\theta$
    	 2. $\textbf{end for}$
    7. $\textbf{end for}$
4. $\textbf{until}$ convergence


Original DDPG algorithm Pseudocode 
=============================

1. Randomly initialize critic network $Q(s, a|\theta^Q)$ and actor $\mu(s|\theta^\mu)$ with weights $\theta^Q$ and $\theta^\mu$
2. Intialize target network $Q\prime$ and $\mu\prime$ with weights $\theta^{Q\prime} \leftarrow \theta^Q, \theta^{\mu\prime} \leftarrow \theta^\mu$
3. Intialize replay buffer $R$
4. $\textbf{for}$ episode = 1, M $\textbf{do}$
    1. Initialize a random process $\mathcal{N}$ for action exploration
    2. Receive intial observation state $s_1$
    3. $\textbf{for}$ $t=1,$ T $\textbf{do}$
        1. Select action $a_t = \mu(s_t|\theta^\mu) + \mathcal{N}_t$ according to the current policy and exploration noise
        2. Execute action $a_t$ and observe reward $r_t$ and observe new state $s_{t+1}$
        3. Store transition $(s_t, a_t, r_t, s_{t+1})$ in $R$ 
        4. Sample a random minibatch of $N$ transitions $(s_i, a_i, r_i, s_{i+1})$ from $R$
        5. Set $y_i = r_i + \gamma Q\prime(s_{i+1}, \mu\prime(s_{i+1} |\theta^{\mu\prime}) |\theta^{Q\prime})$
        6. Update critic by minimizing the loss: $L = \frac{1}{N} \sum_i(y_i - Q(s_i, a_i|\theta^Q))^2$
        7. Update the actor policy using the sampled policy gradient:<br />
        $\nabla_{\theta^\mu}J \approx \frac{1}{N} \sum_i \nabla_a Q(s, a|\theta^Q)$ $\textbar_{s=s_i, a=\mu(s_i)}$ $\nabla_{\theta^{\mu}} \mu(s|\theta^\mu)|_{s_i}$
	8. Update the target networks:<br />
	$\theta^{Q\prime} \leftarrow \tau \theta^Q + (1 + \tau)\theta^{Q\prime}$<br />
        $\theta^{\mu\prime} \leftarrow \tau \theta^\mu + (1 + \tau)\theta^{\mu\prime}$
    4. $\textbf{end for}$
5. $\textbf{end for}$


