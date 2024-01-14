# Flappy Bird Reinforcement Learning :bird: :video_game: :robot:

## Prezentarea temei de proiect 

Tema proiectului este implementarea unui mediu de tip Flappy Bird și a 3 algoritmi de Reinforcement Learning asupra acestui mediu:
- Q-Learning cu deep Q network (DQN)
- SARSA (State-Action-Reward-State-Action)
- PPO (Proximal Policy Optimization)

## Descrierea mediului

Mediul este implementat în Python (cu biblioteca PyGame) și este identic cu jocul original de Flappy Bird. Jucătorul controlează o pasăre care trebuie să se strecoare printre stâlpii verticali (obstacole). Singura mișcare pe care o poate face pasărea este să sară (deplasare pe verticală). Mișcarea hărții pe orizontală este automată și constantă, până la finalizarea jocului. De asemenea, dacă utilizatorul nu apasă pe spațiu pentru a sări, pasărea va cădea (efect de "gravitație"). Jocul se termină atunci când pasărea nu se strecoară prin spațiul gol și lovește un obstacol sau atinge marginea mediului de joc. Pentru fiecare "strecurare", jucătorul câștigă un punct.

## Descrierea agentilor

Pentru fiecare dintre cei 3 algoritmi pe care i-am implementat, avem câte un agent unic. Acestora le corespunde câte o politică distinctă.

### 1. DQN

Ideea de bază este ca agentul să învețe o politică, astfel încât să fie maximizată recompensa totală (scorul). Q-Function este o funcție care estimează valoarea de a face o anumită acțiune într-o stare specifică. În general, algoritmii de Q-Learning folosesc o tabelă pentru a reprezenta această funcție (Q-Table). Pentru mediile mai complexe, această tabelă devine foarte mare și algoritmul devine ineficient. Astfel, pentru a rezolva această problemă, folosim o rețea neuronală pentru a aproxima Q-Function. Deciziile agentului se bazează pe această aproximare.

Pentru a păstra echilibrul între eplorare și exploatare, algoritmul folosește o strategie numită "ε-Greedy". Această strategie alege o acțiune aleatoare cu probabilitatea ε, și în rest alege acțiunea cu cea mai mare valoare Q.

DQN este un algoritm off-policy, adică agentul învață o politică diferită de cea pe care o folosește pentru a explora mediul. În cazul nostru, agentul folosește o politică ε-Greedy pentru a explora mediul, dar învață o politică care alege întotdeauna acțiunea cu cea mai mare valoare Q (o strategie de tip Greedy).


### 2. SARSA

SARSA este un algoritm de tip on-policy, acesta luându-și deciziile pe baza aceleiași politici pe care o folosește pentru a învața. Abordarea SARSA este de tip Temporal Difference (TD), care învață prin actualizarea funcției Q pe baza "diferenței temporale" dintre recompensa prezisă și cea observată și așa el poate să maximizeze mult mai bine recompensa. Această actualizare se face după fiecare pas dintr-o pereche stare-acțiune la următoarea, nu doar la final de episod. Asemănător cu DQN, SARSA folosește tot o strategie ε-Greedy pentru a alege acțiunile în mod aleatoriu, cu probabilitatea ε, iar în rest urmărește valoarea cea mai bună de pe politica actuală.

Principala caracteristică a SARSA este că este un algoritm de învățare mai sigur, în comparație cu DQN. Asta este un avantaj mare în mediile unde anumite acțiuni pot duce la consecințe negative. Din punct de vedere al timpului, SARSA este aproape optim, deși poate converge mai încet decât DQN în anumite medii pentru că prioritizează siguranța. Performanța depinde de echilibrul pe care îl găsește între explorare și exploatare, precum și de modul în care politica este actualizată în timp.


### 3. PPO

PPO este de asemenea un algoritm off-policy. Acesta face parte din categoria metodelor de policy gradient, pentru că el își parametrizează direct politica. Spre deosebire de DQN și SARSA (algoritmi bazați pe valori), obiectivul principal al acestui algoritm este să învețe o politică în care sunt mapate stările cu acțiuni, în așa fel încât să se maximizeze recompensa cumulativ. 

Pentru a îmbunătăți politica și în același timp să nu fie alterată semnificativ, este folosită o funcție "obiectiv" (objective function) ce compară politica nouă (obținută prin modificările la un anumit pas) de cea curentă. Acest raport este controlat printr-un mecanism de "clipping", și are ca efect o antrenare mai stabilă și previne decizii mai riscante luate de agent. 

Acest algoritm este foarte performant chiar și în medii mai complexe, pentru că este capabil să facă decizii mari și în același timp și sigure, prevenind schimbări majore în politică ce are duce la o antrenare instabilă. 
