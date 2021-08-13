# An explanation for HC HM MM HI types of sentences in the Pascal 50S dataset.

Please look into the new_data field in the file `pair_pascal.mat`,
it can be used to infer the class of each sentence in the "pair" of candidates. 

*1-5 are machine generated sentences*
*6 are human generated sentences*
*7 are random human sentences*

---
So a row in new_data which says 
4 6 -- means it is a HM pair where first in the pair is machine and second sentence is human
6 7 -- means it is a HI pair where the first in the pair is human and second sentence is  machine generated 
