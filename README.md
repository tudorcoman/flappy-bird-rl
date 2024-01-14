# Flappy Bird Reinforcement Learning :bird: :video_game: :robot:

## Prezentarea temei de proiect 

Tema proiectului este implementarea unui mediu de tip Flappy Bird și a 3 algoritmi de Reinforcement Learning asupra acestui mediu:
- Q-Learning cu deep Q network (DQN)
- SARSA (State-Action-Reward-State-Action)
- PPO (Proximal Policy Optimization)

## Descrierea mediului

Mediul este implementat în Python (cu biblioteca PyGame) și este identic cu jocul original de Flappy Bird. Jucătorul controlează o pasăre care trebuie să se strecoare printre stâlpii verticali, care reprezintă obstacolele. Singura mișcare pe care o poate face pasărea este să sară (deplasare pe verticală), mișcarea orizontală fiind automată și constantă, până la finalizarea jocului. Jocul se termină atunci când pasărea nu se strecoară prin spațiul gol și lovește un obstacol sau atinge marginea mediului de joc. Pentru fiecare "strecurare", jucătorul câștigă un punct.

## Descrierea agentilor

Pentru fiecare dintre cei 3 algoritmi pe care i-am implementat, avem câte un agent unic. Acestora le corespunde câte o politică distinctă.

### 1. DQN

Politica: Ideea de baza este ca agentul să învețe o politică, aceasta maximizând recompensa totală.

### 2. SARSA

### 3. PPO