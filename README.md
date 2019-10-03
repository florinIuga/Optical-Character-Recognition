# Optical-Character-Recognition
Introduction to Machine Learning algorithms and concepts.

Name: Florin-Eugen Iuga

Grupa, Seria: 312CA
Data începerii temei: 12-05-2018
Data finalizării temei: 19-05-2018


Probleme întâmpinate:
 -> în general, n-am întâmpinat probleme, doar câteva dificultăți în
    înțelegerea algoritmului.

Algoritmul utilizat:

-> algoritmul a presupus combinarea a mai multor arbori de decizie pentru a da o
   predicție mai bună. Primul pas pentru această combinare a fost antrenarea / 
   generarea fiecărui arbore prin primirea unor teste cu rezultat cunoscut.
   Apoi, am realizat etapa de prezicere, unde codul a trebuit să decidă ce
   cifră este reprezentată de fiecare imagine primită ca parametru.

 -> în esență, am urmărit scheletul propus și am implementat
    funcțiile necesare algoritmului, respectiv cele 5 funcții
    punctate de checker, pe care ulterior le-am apelat în
    cele 6 metode care conțin logica acestuia: train(), make_leaf(),
    get_entropy_by_indexes(), find_best_split(), 2 X predict().
    
 -> make_leaf(): 
     - folosește un vector de frecvență pentru a afla
     clasa care apare cel mai des din setul de date primit.
  
 -> find_best_split():
     - am calculat media aritmetică, drept o valoare de split cât mai bună.
     - am calculat information gain pentru fiecare atribut (coloană dată) în
     parte.
     - cel mai mare information gain dă indicele de split (indicele
     atributului corespunzător) și valoarea de split (media aritmetică pe acea
     coloană / a acelui atribut).
     - pasul de mai sus se realizează doar dacă splitul este valid (rezultă 2
     mulțimi nevide în urma acestuia).

->train()
     - verificăm dacă toate testele au aceeași clasă => nodul devine frunză.
     - dacă nu, se caută un split în functie de un vector cu dimensiuni
     aleatoare.
     - verificăm dacă s-a găsit un split, dacă nu => nodul devine frunză.
     - daca da, devine nod de decizie, se realizează splitul și se antrenează
     copiii săi cu cele două mulțimi rezultate.

->Node::predict()
     - prezice rezultatul prin parcurgerea arborelui de decizie.

->get_entropy_by_indexes()
     - folosim un vector de frecvență și calculăm fiecare probabilitate, iar
     apoi utilizăm formula de la entropie.

->get_split_as_indexes()
     - obținem indecșii sample-urilor din cele 2 subseturi obținute în urma separării

->random_dimensions()
     - folosim un random device pentru a obține niște dimensiuni random ale lui
     samples.

->get_random_samples()
    - întoarce elemente random, diferite din samples.

->RandomForest::predict()
    - acesta reprezintă algoritmul Random Forest propriu zis
    - parcurge fiecare arbore de decizie și întoarce cea mai probabilă prezicere.
	
Alte precizări:
    -> Cerința temei a fost foarte interesantă, ne-a ajutat să ne facem o idee
       despre ce înseamna Machine Learning, despre cum sunt folosite
       structurile de date practic și în acest scop.
