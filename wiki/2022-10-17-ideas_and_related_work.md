## Ideje
#### Augmencacija generiranjem novih dokumenata (npr. na temelju argumenata)
  - npr. hipoteza - konkretni tipovi i vrijednosti argumenata su vrlo relevantni za određivanje tipa događaja
  - ako je tome tako, tj. ako argumenti dovoljno dobro određuju tip eventa, onda bismo uvjetnim generiranjem dokumenata mogli postići očuvanje tipa događaja
  - takvo generiranje bi moglo biti bilo što od jednostavne supstitucije argumenata (ovo su ljudi već radili), do nekakvog end-to-end conditional generation modela (malo previše high-risk)
  - alternativno, argumenti mogu biti augmentirani putem back-translationa (+ opcionalne augmentacije u target jeziku)
  - nadalje, budući da su veliki jezični modeli dobri u few shot learningu, opcija je i nekakav prompting GPT3 modela (ovo su ljudi već radili, ali ne specifično za document-level event etraction)
    - glavni nedostatak je, naravno, što koristimo ogromni model, i taj model se trenutno nudi samo kao plaćeni service
    - inference bi mogao biti još skuplji ako uzmemo u obzir da je docee document-level dataset i da su primjeri relativno velike duljine
    - općenito, ovdje bi vjerojatno trebalo posvetiti vremena i prompt engineeringu, što je samo po sebi već velika tema
    - no, npr. bilo bi zanimljivo detaljnije pogledati što bismo sve mogli dobiti iz takvog velikog modela, na temelju što manje inputa
      - generiranje argumenata nekog specifičnog tipa
      - generiranje cijelih dokumenata na temelju oznake i npr. argumenata
  - ne mora fokus biti na argumentima, pitanje je koliko su oni uopće relevantni za određivanje tipa događaja
    - ako sam dobro skužio docee, iako svaki tip događaja ima određene moguće tipove argumenata, ne pojavljuju se svi tipovi u svakom primjeru
      - također - neki tipovi (npr. date) se pojavljuju u više različitih tipova događaja
    - druga opcija bi bila istražiti možemo li nekako mjeriti relevantnost pojedinih dijelova dokumenata (rečenica, spanova), i iskoristiti tu relevantnost za augmentaciju
      - jedna dimenzija - relevantnost (relevantan vs nerelevantan), i kako bi augmentacija različitih dijelova utjecala na event detection
        - pitanje je i kako mjeriti relevance - jedna ideja bi bila brisanje dijela dokumenta i proučavanje utjecaja toga na rezultat klasifikacije
      - druga dimenzija - vrsta augmentacije (dodavanje, brisanje, modificiranje)
      - npr. nerelevantni dijelovi bi se potencijalno mogli agresivno augmentirati, i rizik promjene labele bi bio relativno mali, ali je opet pitanje koliki bi bio utjecaj augmentacije

#### Cross-lingual strukturna augmentacija
  - ideja je bazirana na paperu koji je augmentirao tekst tako da su modificirali dependency tree
  - skužili su da im metoda bolje radi u jezicima koji imaju jače strukture (npr. više padeža)
  - budući da je hrvatski visoko strukturiran jezik, takva vrsta augmentacije mogla bi biti jako uspješna
  - nadovezujući se na prijašnju ideju, možda bi se isplatilo fokusirati na relevantnost 
    - npr. augmentacija najmanje relevantnih rečenica pomoću back translationa + strukturne augmentacije
  - osim navedenog, vjerojatno postoji još načina kako se ovo da iskombinirati s cross-lingual pristupom, pogotovo ako na kraju krajeva želimo model koji radi dobro na target languageu (hrvatskom)

## Paperi
### Conditional BERT Contextual Augmentation 
  - link: https://link.springer.com/chapter/10.1007/978-3-030-22747-0_7
  - sažetak:
    - augmentacija na nivou tokena, pomoću BERTa
    - zamjena tokena maskiranjem i predikcijom (kao MLM)
    - da izbjegnu problem utjecaja na oznaku primjera, rade augmentaciju uvjetovanu oznakom primjera
    - reformuliraju segment embeddinge u BERTu kao label embeddings, i to koriste kako bi uvjetovali generiranje oznakom primjera
    - rezultati ukazuju na napredak u domeni klasifikacije teksta u odnosu na BERT augmentaciju bez uvjetovanja
  - relevantnost:
    - npr. mogli bismo mijenjati argumente u dokumentu, uvjetovane tipom argumenta
    - općenito ako će nam trebati bilo kakvo uvjetno generiranje dijelova teksta, ovaj paper pokazuje da je tako nešto donekle izvedivo
  - problemi:
    - nisu dovoljno dobro objasnili kako koriste segment embeddings za labele (kažu da treba retrenirati ako je broj labela > 2, ali nisu objasnili kako točno)
    - segment embeddings su specifični za BERT, pa nije jasno kako ovakvu metodu generalizirati na ostale transformere
### GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation
  - link: https://aclanthology.org/2021.findings-emnlp.192/
  - sažetak:
    - generiraju augmentirane primjere promptanjem GPT3 modela
    - rezultati ukazuju na poboljšanja, pogotovo u low-data režimu
    - napravljen je ablation study za mnoge aspekte metode
  - relevantnost:
    - ako bismo išli na augmentaciju pomoću generiranja, GPT3 prompting bi nam vjerojatno dao najbolje rezultate (čisto zbog skale modela), a bio bi relativno jednostavan s implementacijske strane (u usporedbi s osmišljavanjem i treniranjem vlastitog generativnog modela)
    - jednostavnost u implementaciji bi nam omogućila da se više fokusiramo na različite aspekte augmentacije (ako bismo htjeli takvo nešto)
  - problemi:
    - inference na GPT3 se plaća
    - primjeri iz docee dataseta su relativno velike duljine, što sigurno pogoršava financijske aspekte
    - koristimo end-to-end black box model
    - trebali bismo potrošiti vrijeme na prompt engineering, iako to bi moglo biti zanimlijvo samo po sebi
### CrudeOilNews: An Annotated Crude Oil News Corpus for Event Extraction
  - link: https://ui.adsabs.harvard.edu/abs/2022arXiv220403871L/abstract
  - sažetak:
    - novi sentence-level dataset za event extraction
    - nisam čitao sve detaljno, pa navodim što mi je zapelo za oko
    - kako bi poboljšali performanse na klasama s malo primjera, autori augmentiraju podatke na dva načina
    - prvi je trigger word replacement (to nama nije zanimljivo na document-levelu)
    - drugi je event argument replacement, na način da zamjenski argument mora imati isti entity type, i isti argument role
    - augmentacijom postižu značajno bolje rezultate
  - relevantnost:
    - pokazuju da je argument replacement obećavajuća metoda augmentacije u event extractionu
  - problemi:
    - nisu napravili ablaciju na dva tipa augmentacije, pa nije baš jasno koji tip donosi koliko poboljšanja
### Data Augmentation via Dependency Tree Morphing for Low-Resource Languages 
  - link: https://aclanthology.org/D18-1545/
  - sažetak:
    - augmentacija koja čuva značenje, bazira se na stablima ovisnosti
    - rade cropping i rotaciju u nad stablima kako bi dobili drugačije rečenice s istim značenjem
    - poboljšanje u POS taggingu (koristeći Bi-LSTM)
    - poboljšanja su najveća u strukturiranim jezicima (ne znam lingvističku terminologiju, autori navode "rich case-marking systems")
  - relevantnost:
    - mogli bismo raditi takvu augmentaciju u target jeziku (npr. hrvatski)
  - problemi:
    - evaluacija samo na jednom modelu i jednom tasku
    - nije jasno jel bi ova vrsta augmentacije radila s transformerima
    - nije jasno kako dobili stabla ovisnosti za docee, pogotovo u target jeziku
### Exploring Pre-trained Language Models for Event Extraction and Generation 
  - link: https://aclanthology.org/P19-1522/
  - sažetak:
    - predlažu poboljšanje za, između ostalog, argument identification na zadatku sentence-level event extractiona
    - provode augmentaciju u dva koraka
    - prvo provode zamjenu argumenata, na način da zamjenski argument mora imati isti role i biti što sličniji (cosine similarity)
    - nakon toga mijenjaju "adjunct tokens" (nisam uspio skužiti što ovo znači u kontekstu papera), koristeći BERT i maskiranje
    - novi primjeri su rangirani s obzirom na parametriziranju kombinaciju perplexityja i prosječne kosinusne udaljenosti od originalnog dataseta
    - rezultati su state-of-the-art (u 2019.)
  - relevantnost:
    - rangiranje augmentiranih primjera se može lako primijeniti na document-level
    - a možda i poboljšati
    - opet princip argument replacementa, kojeg bi bilo fora iskoristiti na pametan način u document-level event extractionu
  - problemi:
    - nisu radili ablaciju nad dva koraka augmentacije, nije jasno kako koji utječe na rezultate
    - nisu dovoljno dobro objasnili drugi korak augmentacije, što točno misle pod adjunct tokens i kako su ih točno birali i mijenjali
