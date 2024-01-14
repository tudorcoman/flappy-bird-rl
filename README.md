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

Ideea de bază este ca agentul să învețe o politică, aceasta maximizând recompensa totală. Q-Function este o funcție care estimează valoarea de a face o anumită acțiune într-o stare specifică. În general, algoritmii de Q-Learning folosesc o tabelă pentru a reprezenta această funcție (Q-Table). Pentru mediile mai complexe, această tabelă devine foarte mare și algoritmul devine ineficient. Astfel, pentru a rezolva această problemă, folosim o rețea neuronală pentru a aproxima Q-Function. Deciziile agentului se bazează pe această aproximare.

Pentru a păstra echilibrul între eplorare și exploatare, algoritmul folosește o strategie numită "ε-Greedy". Această strategie alege o acțiune aleatoare cu probabilitatea ε, și în rest alege acțiunea cu cea mai mare valoare Q.

DQN este un algoritm off-policy, adică agentul învață o politică diferită de cea pe care o folosește pentru a explora mediul. În cazul nostru, agentul folosește o politică ε-Greedy pentru a explora mediul, dar învață o politică care alege întotdeauna acțiunea cu cea mai mare valoare Q (o strategie de tip Greedy).


### 2. SARSA

SARSA este un algoritm de tip on-policy, acesta luându-și deciziile pe baza aceleiași politici pe care o folosește pentru a învața. Abordarea SARSA este de tip Temporal Difference (TD), care învață prin actualizarea funcției Q pe baza "diferenței temporale" dintre recompensa prezisă și cea observată și așa el poate să maximizeze mult mai bine recompensa. Această actualizare se face după fiecare pas dintr-o pereche stare-acțiune la următoarea, nu doar la final de episod. Asemănător cu DQN, SARSA folosește tot o strategie ε-Greedy pentru a alege acțiunile aleatoriu cu probabilitatea ε, iar în rest urmărește politica actuală cea mai bună.

Principala caracteristică a SARSA este că este un algoritm de învățare mai sigur, pentru că învață într-un mod "mai safe" comparativ cu DQN. Asta este un avantaj mare în mediile unde anumite acțiuni pot duce la consecințe negative. Din punct de vedere al timpului, SARSA este aproape optim, deși poate converge mai încet decât DQN în anumite medii pentru că prioritizează siguranța. Performanța depinde de echilibrul pe care îl găsește între explorare și exploatare, precum și de modul în care politica este actualizată în timp.


### 3. PPO

PPO (Policy Gradient Agent) este algoritmul care nu are nevoie de un model, pentru că el își parametrizează direct politica. Rețeaua neuronală returnează o distribuție de probabilități ale acțiunilor, iar deciziile sunt luate pe baza acestui rezultat. Rețeaua neuronală își actualizează parametrii pentru a crește probabilitatea acțiunilor care au adus și vor aduce recompense mai mari.

Ce este specific PPO reprezintă faptul că are o abordare mai sofisticată pentru a actualiza parametrii politicii, încercând să vizeze deciziile pe termen lung și sigure. Astfel, PPO încearcă să mențînă un echilibru între explorare și exploatare prin schimbări limitate ale politicii la fiecare update, prevenind schimbările drastice care pot duce la o învățare instabilă, deci el prioritzează siguranța în detrimentul timpului de învățare.