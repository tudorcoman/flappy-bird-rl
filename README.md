# Flappy Bird Reinforcement Learning :bird: :video_game: :robot:

## Prezentarea temei de proiect 

Tema proiectului este implementarea unui mediu de tip Flappy Bird si a 3 algoritmi de Reinforcement Learning asupra acestui mediu:
- Q-Learning cu deep Q network (DQN)
- SARSA (State-Action-Reward-State-Action)
- PPO (Proximal Policy Optimization)

## Descrierea mediului

Mediul este implementat in Python (cu biblioteca PyGame) si este identic cu jocul original de Flappy Bird. Jucatorul controleaza o pasare care trebuie sa se strecoare printre stalpii verticali, care reprezinta obstacolele. Singura miscare pe care o poate face pasarea este sa sara (deplasare pe verticala), miscarea orizontala fiind automata si constanta, pana la finalizarea jocului. Jocul se termina atunci cand pasarea nu se strecoara prin spatiul gol si loveste un obstacol sau atinge marginea mediului de joc. Pentru fiecare "strecurare", jucatorul castiga un punct.

