## STROOP EFFECT-The test of a perceptual phenomenon.
Background Information In a Stroop task:
Participants are presented with a list of words, with each word displayed in a color of ink. The participant's task is to say out loud the color of the ink in which the word is printed. The task has two conditions: a congruent words condition, and an incongruent words condition. In the congruent words condition, the words being displayed are color words whose names match the colors in which they are printed: for example RED, BLUE. In the incongruent words condition, the words displayed are color words whose names do not match the colors in which they are printed: for example PURPLE, ORANGE. In each case, we measure the time it takes to name the ink colors in equally-sized lists. Each participant will go through and record a time from each condition.

With this time recorded for congruent words and Incongruent words as the data, the following Statistical Alternative Hypothesis, *"Two-Tailed dependant sample t Test at 95% Confidance Interval"* is validated 

Using the R language, simple Histograms are representing the data distribution.
A Box plot is used to show the significant difference in the medians of time taken in reading under the two conditions (Congruent and Incongruent). The time taken to read the incongruent words by each person is more than the time taken to read the congruent words.

The values of parameters to test the following Null Hypothesis is calculated:

Hypothesis

H0 : $\muc$ - $\mui$ = 0 ;
HA : $\muc$ - $\mui$ != 0

The following formulae are used to calculate:
1) Calculate the difference between means.
     mean_diff = $\mu~c~$ - $\mu~i~$  
2) Calculate the standard error of Means
     EM = $\s~d~$/$\sqrt(n)$, where $\s~d~$ = standard deviation of the differences.
3) Calculate the *t-statistical value* : 
     t = mean_diff / SEM 
     
These analysis lead to the following decision:

Clearly, when t- statistical value is lying outside the t-critical values, beyond the $\alpha$ region, we have proved the Alternative Hypothesis, accepting the fact that there is a significant difference between the time taken to read the congruent and Incongruent words. In other words we reject the null, which means $\muc$ - $\mui$ != 0

So, Yes. The results were as expected in the beginning of the analysis.