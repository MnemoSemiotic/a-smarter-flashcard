[Google Presentation](https://docs.google.com/presentation/d/1382PGj1Ljha43d8BnQAKAfSGJM_bGphilbakYj5520A/edit?usp=sharing)

# In Search of a Smarter Flashcard
## Discerning Topics, Pedagogical Implications
#### Tovio Roberts, Capstone 2 @Galvanize


### **GOALS:**
- Clean flash card pool in a way that can be generalized to new card content
- Topic Model in a ‘reasonable’ way so as to enable simple similarity selection in an application
- Discuss discerning strong and weak subjects for a user
  - How to deliver the most "helpful" study materials
  - Associative Database --> Extrapolative Learning

### **DATA**
3 collections of ~12,000 each, flash cards, ~36,000 total.  These are divided into 3 general categories:
- Data Science
- Biology
- History

#### Each "card" is composed of a question and an answer.

The data sets are compiled from three sources:
- AnkiWeb
- Quizlet
- My own collection

### **PROJECT PROGRESSION Minimum Viable Product:**
1. Create data cleaning pipeline.
    * Strip html from cards
    * Standardize, modify or ignore formulas that are not consistent across cards.
    * Modify entries that lead to erroneous topics.
2. Explore NLP strategies to allow for meaningful clustering
    * Stem, Lemmatize, Stopwords
    * Count Vector
    * TF-IDF Vector
3. Use Clustering to analyze topics within a single subject corpus.
    * Provide list of “quintessential” words for each topic, most-common words per category.
    * User chosen categories become target labels.
4. Apply same Topic modeling to the full pool of cards


### *Improvement 1: Provide a simple API for flashcards*
1. Build topic distribution table when new cards are added
2. Retrieve flashcard
3. Update success table for flashcard user

### *Improvement 2: Provide an Interface for Card Review*
1. Swipe Left/Swipe Right simple front end.
2. Update success/fail.
3. Discern “Strong” and “Weak” topics.

### *Improvement 3: Smart Flashcard Delivery*
1. Incorporate Spaced Repetition and randomness settings into reviews.
2. Use similarity metrics to discern “Weak” and “Strong” topics, based on card review successes.
3. Deliver review cards as a function of spaced repetition, strength, and similarity.



--------------------------------------------------------------------------------
## Topic Modeling using TF Vector


<link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">


<div id="ldavis_el395941121577884645859624963"></div>
<script type="text/javascript">

var ldavis_el395941121577884645859624963_data = {"mdsDat": {"Freq": [36.58741390894549, 32.75353805010696, 30.659048040947546], "cluster": [1, 1, 1], "topics": [1, 2, 3], "x": [0.17456565813311062, -0.3053439361270054, 0.1307782779938947], "y": [0.22291453540541908, 0.022380982174982595, -0.24529551758040172]}, "tinfo": {"Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3"], "Freq": [3023.0, 2120.0, 1779.0, 1334.0, 1170.0, 1159.0, 1240.0, 981.0, 911.0, 848.0, 1045.0, 943.0, 3889.0, 802.0, 869.0, 1778.7264356906398, 942.4313002366583, 867.838894606125, 868.763905757893, 707.8537753391007, 703.9083842286541, 711.8506472607096, 742.1304842441137, 666.3451751931713, 668.9107366540281, 655.3615327998862, 601.279533810975, 551.8764083632428, 554.126825253619, 515.5582018754346, 2106.197684226375, 1237.1849797087648, 796.0199044224518, 645.6950351614267, 1036.7667960116457, 1017.5929722634546, 2828.4737782906664, 1237.8953605470426, 1508.730252707349, 860.8916384147873, 762.1713152170591, 947.0764237508569, 900.0902302433096, 1158.7800466138192, 980.9205396458381, 910.3748450833409, 801.2173809978154, 660.7131297629714, 648.4488892950387, 644.5598330088555, 643.4415480368018, 613.8839895152454, 608.0271610825877, 615.7190808775716, 1332.9349088496992, 600.9976086663469, 594.0567944248631, 544.9409331656022, 566.6633980524908, 719.585239371059, 642.615099188916, 3023.178298968958, 1169.6184672551317, 669.9102052553443, 646.8033386643989, 637.5594810683631, 582.6068814669138, 553.654633748497, 554.5322112707001, 504.129985730573, 480.80100663909894, 465.6098552829235, 847.1558733445295, 431.0056170051796, 447.86529428946614, 424.7773100698899, 492.03751876902425, 886.8025615365102, 776.5761505377475, 906.5073699897132, 1140.5372796000372, 829.7101705067848, 604.8719546181944, 655.4311829777018], "Term": ["cell", "data", "model", "state", "protein", "war", "variabl", "empir", "peopl", "produc", "valu", "distribut", "use", "power", "featur", "model", "distribut", "test", "featur", "algorithm", "linear", "train", "matrix", "theta", "error", "text", "sampl", "frac", "measur", "random", "data", "variabl", "learn", "problem", "valu", "mean", "use", "set", "function", "number", "point", "differ", "caus", "war", "empir", "peopl", "power", "amask", "china", "roman", "did", "govern", "nation", "style", "state", "trade", "world", "polit", "display", "new", "rule", "cell", "protein", "plant", "dna", "receptor", "blood", "gene", "bind", "water", "molecul", "speci", "produc", "antigen", "virus", "cells", "contain", "organ", "activ", "type", "caus", "form", "infect", "use"], "Total": [3023.0, 2120.0, 1779.0, 1334.0, 1170.0, 1159.0, 1240.0, 981.0, 911.0, 848.0, 1045.0, 943.0, 3889.0, 802.0, 869.0, 1779.338632163844, 943.0601120128987, 868.4728951389319, 869.4737584102062, 708.460024601197, 704.5152917217724, 712.4816979880339, 742.7954747074506, 666.9446370000635, 669.519254842425, 656.0004464047882, 601.8912801459609, 552.476416031861, 554.7495571641165, 516.1710773129425, 2120.5834945944707, 1240.5297567099203, 796.970502953179, 646.6085468798886, 1045.609050919083, 1077.2558639712552, 3889.9339785799602, 1433.6668344457785, 2074.423611817141, 948.003825645194, 809.0744844451722, 1305.8988408034197, 2055.9150788431066, 1159.5699645083694, 981.7801093405245, 911.1942340082804, 802.1190849143151, 661.5014984416127, 649.2437252447666, 645.3525329387252, 644.2682745883309, 614.6817722466835, 608.8223888715229, 616.5261356410812, 1334.7001573823711, 601.7952217827113, 594.8789938165248, 545.7322965748851, 568.7317279331039, 1098.289052302655, 859.7747859035018, 3023.8663909451884, 1170.3046958819534, 670.6039712461796, 647.4883782158549, 638.2421378641178, 583.3088382207563, 554.341406205109, 555.2253297559221, 504.84431689954545, 481.4836180600013, 466.3338212358507, 848.4919048123463, 431.6957396584322, 448.58968551819197, 425.47907794264603, 494.4117163708682, 1026.7449280721426, 876.2741486655874, 1181.0165641618387, 2055.9150788431066, 1390.3973185342015, 1029.9157874989526, 3889.9339785799602], "loglift": [15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0051, 1.0048, 1.0047, 1.0046, 1.0046, 1.0046, 1.0046, 1.0046, 1.0046, 1.0046, 1.0045, 1.0044, 1.0044, 1.0043, 1.0043, 0.9987, 1.0028, 1.0043, 1.0041, 0.997, 0.9485, 0.6868, 0.8586, 0.6871, 0.9091, 0.9457, 0.6842, 0.1795, 1.1155, 1.1153, 1.1153, 1.115, 1.115, 1.1149, 1.1149, 1.1149, 1.1149, 1.1149, 1.1148, 1.1148, 1.1148, 1.1148, 1.1147, 1.1125, 0.6933, 0.825, 1.182, 1.1817, 1.1812, 1.1812, 1.1812, 1.181, 1.181, 1.181, 1.1808, 1.1808, 1.1807, 1.1807, 1.1806, 1.1806, 1.1806, 1.1774, 1.0357, 1.0615, 0.9177, 0.593, 0.666, 0.65, -0.5986], "logprob": [15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -4.5697, -5.2048, -5.2873, -5.2862, -5.4911, -5.4967, -5.4854, -5.4438, -5.5515, -5.5477, -5.5681, -5.6542, -5.74, -5.7359, -5.8081, -4.4007, -4.9327, -5.3737, -5.583, -5.1094, -5.1281, -4.1058, -4.9321, -4.7343, -5.2953, -5.4171, -5.1999, -5.2508, -4.8875, -5.0541, -5.1288, -5.2565, -5.4493, -5.468, -5.474, -5.4758, -5.5228, -5.5324, -5.5198, -4.7475, -5.544, -5.5556, -5.6419, -5.6028, -5.3639, -5.4771, -3.8625, -4.8121, -5.3694, -5.4045, -5.4189, -5.509, -5.56, -5.5584, -5.6537, -5.7011, -5.7332, -5.1346, -5.8104, -5.772, -5.825, -5.678, -5.0889, -5.2216, -5.0669, -4.8373, -5.1555, -5.4715, -5.3912]}, "token.table": {"Topic": [1, 3, 1, 2, 3, 3, 3, 1, 2, 3, 3, 3, 2, 1, 2, 3, 1, 3, 2, 1, 2, 3, 2, 3, 1, 3, 1, 2, 1, 1, 1, 2, 3, 1, 1, 2, 3, 3, 2, 1, 3, 1, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 2, 3, 1, 3, 2, 3, 2, 3, 1, 2, 3, 2, 1, 2, 1, 2, 1, 2, 3, 3, 1, 3, 2, 1, 2, 1, 1, 2, 3, 3, 1, 2, 3, 2, 1, 1, 1, 2, 1, 1, 3, 1, 2, 3, 1, 3, 1, 3, 3, 2, 3, 2], "Freq": [0.11297834148224015, 0.886708801330309, 0.9993506696422907, 0.9992418786007375, 0.9983883564406202, 0.999594165208527, 0.9994705408172824, 0.43776127198135206, 0.007296021199689201, 0.5549840125896919, 0.9997134823986328, 0.9988740270262816, 0.9980843476857667, 0.00202260578964492, 0.00202260578964492, 0.9951220485053006, 0.9931228859266117, 0.006601956506634645, 0.9980314495710016, 0.7251710242865238, 0.034459024385315284, 0.23968165850230408, 0.9969551058116674, 0.0035165964931628477, 0.9988758807637024, 0.9992457343910934, 0.001018558015675948, 0.9992054133781051, 0.9992244362821989, 0.9994551205190224, 0.19203144053922616, 0.211450350256676, 0.5969516690919764, 0.9991376717303466, 0.7274309795761317, 0.06748862633575774, 0.20535824870737712, 0.9993841228504899, 0.9988908533204236, 0.41265509778432463, 0.5874266686106269, 0.9987822598834175, 0.0012547515827681125, 0.9992685868882801, 0.9989290797607997, 0.9449936955991021, 0.054768789823523595, 0.9986488368409914, 0.9998096864993977, 0.9989955669479476, 0.9986492138157941, 0.2230742439673208, 0.655565125128453, 0.121097446725117, 0.908224182971012, 0.09177178155456218, 0.13537929061017323, 0.8638951854044867, 0.9986893749282992, 0.9990993622583874, 0.9418168718082195, 0.04820322572246793, 0.009887841173839574, 0.9986581395686471, 0.0012466976771994189, 0.9986048394367345, 0.9990588635383415, 0.0015465307485113646, 0.001178561627198036, 0.001178561627198036, 0.9982416982367366, 0.9997396439721847, 0.9996685647056531, 0.9996206175528802, 0.9994537358718965, 0.2523916769342842, 0.7478702685195611, 0.9985192007670476, 0.8635200105459518, 0.06347360336000131, 0.07323877310769382, 0.9992841582131744, 0.0007492319488155386, 0.998726187771113, 0.0007492319488155386, 0.9991466125916396, 0.9994554865884948, 0.998474930298796, 0.9985836350610562, 0.9986785840865342, 0.999323915281762, 0.23200351994593435, 0.7679824547115418, 0.7270046267038125, 0.10437195135846812, 0.16838332054137098, 0.9917664724578313, 0.008607423579672596, 0.9971546376128197, 0.002418321675698027, 0.9986854679516075, 0.9995084690654168, 0.9983275697650105, 0.9985223989657367], "Term": ["activ", "activ", "algorithm", "amask", "antigen", "bind", "blood", "caus", "caus", "caus", "cell", "cells", "china", "contain", "contain", "contain", "data", "data", "did", "differ", "differ", "differ", "display", "display", "distribut", "dna", "empir", "empir", "error", "featur", "form", "form", "form", "frac", "function", "function", "function", "gene", "govern", "infect", "infect", "learn", "learn", "linear", "matrix", "mean", "mean", "measur", "model", "molecul", "nation", "new", "new", "new", "number", "number", "organ", "organ", "peopl", "plant", "point", "point", "point", "polit", "power", "power", "problem", "problem", "produc", "produc", "produc", "protein", "random", "receptor", "roman", "rule", "rule", "sampl", "set", "set", "set", "speci", "state", "state", "state", "style", "test", "text", "theta", "trade", "train", "type", "type", "use", "use", "use", "valu", "valu", "variabl", "variabl", "virus", "war", "water", "world"]}, "R": 15, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [1, 3, 2]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el395941121577884645859624963", ldavis_el395941121577884645859624963_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
        new LDAvis("#" + "ldavis_el395941121577884645859624963", ldavis_el395941121577884645859624963_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){
         LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el395941121577884645859624963", ldavis_el395941121577884645859624963_data);
            })
         });
}
</script>
