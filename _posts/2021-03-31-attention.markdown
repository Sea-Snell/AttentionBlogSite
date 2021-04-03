---
layout: post
title:  "Approximating How Single Head Attention Learns"
date:   2021-03-31 23:41:24 -0700
categories: Attention
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

**We propose a simple model to approximate how single head attention learns. This allows us to:**

- **Understand why neural networks frequently attend to salient words;**
- **Construct a distribution where an Attention-Seq2Seq architecture cannot even learn to copy the input.**

<div class="links">
  <a href="https://github.com/Sea-Snell/AttentionDynamics"><button class="code-button">code</button></a>
  <a href="https://arxiv.org/pdf/2103.07601.pdf"><button class="paper-button">paper</button></a>
</div>



## Background

Attention mechanism underlies many recent advances in NLP, such as machine translation, language modelling, etc. However, we still do not know much about how attention mechanism is learned, and its behaviors can sometimes be surprising.

For example, while a classifier found by the standard gradient descent attends to salient (key) words, it is also possible to construct a classifier that **attends uniformly to all words but achieves the same test loss**.

Loss alone cannot explain where the classifier attends to, so we need to study how attention is learned at training time. 

Our paper approximates how single head attention learns in a Seq2Seq architecture.

To build up necessary intuitions, our blog will start with something easier: analyzing the optimization trajectory of a simple keyword detection classifier. Later we slowly develop our framework to model more complex situations.


## Warmup: A Toy Classifier

To provide intuitions on how attention learns, we illustrate how a toy classifier learns a toy classification task. We will introduce the task, the loss function, the classifier and its parameters; then we will calculate the gradient, and analyze how the parameters evolve at training time. 

**Task:**
Consider the following simple sequence classification task of keyword detection: each sequence consists of tokens either 0 or 1, and the label is positive if token 1 appears once in the sequence. For example, suppose there are only two following sequences in the training set:

<center style="padding-bottom:2%;">

Sequence 0: {0, 0} $\rightarrow$ Negative

</center>

<center style="padding-bottom:2%;">

Sequence 1: {0, 1} $\rightarrow$ Positive

</center>


**Classifier:**
We consider a unigram gated-classifier $c$. Voab 0 is associated with a classifier weight $\beta_{0}$ and Vocab 1 with a classifier weight $\beta_{1}$.

Additionally, the classifier learns an "attention weight" $0 < p < 1$: how much the classifier should attend to Vocab 1, as opposed to Vocab 0, which recieves attention weight (1 - $p$).

Our classifier $c$ would score the two sequences in the following way:

$$s_{0} = c(\{0, 0\}) = \beta_{0}$$

$$s_{1} = c(\{0, 1\}) = (1-p)\beta_{0} + p\beta_{1}$$

**Loss:**
We want $s_{1}$, the score of the positive sequence, to be as large as possible, and $s_{0}$ to be as small as possible.

Therefore, to train this classifier, we can use gradient descent to minimize the loss $\mathcal{L} = s_{1} - s_{0}$ for these two sequences

$$\mathcal{L} = s_{1} - s_{0} = \beta_{0} - ((1-p)\beta_{0} + p\beta_{1}) = p(\beta_{0} - \beta_{1})$$

Intuitively, we hope that $\beta_{1}$ will become positive since it is associated with positive sequence, $\beta_{0}$ will become negative, and the classifier will learn to attend to the key word and $p$ converges to 1.

**Gradient**:
If we train the parameter with gradient flow (gradient descent with infinitesimal step size), and let the training time be $\tau$, the change of the parameters $\beta_{0}, \beta_{1}$ and $p$ throughout the training time will be governed by the following set of differential equations:

$$\frac{\partial \beta_{0}(\tau)}{\partial \tau} = -\frac{\partial \mathcal{L}}{\partial \beta_{0}} = -p$$

$$\frac{\partial \beta_{1}(\tau)}{\partial \tau} = -\frac{\partial \mathcal{L}}{\partial \beta_{0}} = p$$

$$\frac{\partial p(\tau)}{\partial \tau} = -\frac{\partial \mathcal{L}}{\partial p} = \beta_{1} - \beta_{0}$$


**Dynamics**:
How will these parameters evolve through time? If we assume that $p > 0$, then $\beta_{0}$ will always be decreasing and $\beta_{1}$ increasing.

This is expected, since $\beta_{1}$ is associated with the positive sequence and $\beta_{0}$ the negative one. Interestingly, $p$ will not necessarily increase to 1 - it depends on the value of $\beta_{1} - \beta_{0}$.

Suppose the classifier is uniformly initialized, say $\beta_{0} = \beta_{1}$, $p = 0.5$. According to the above differential equations, initially $p$ will have 0 gradient and remain 0.5, while $\beta_{0}$ will decrease and $\beta_{1}$ will increase; later $\beta_{1} - \beta_{0} > 0$ continues to increase, and $p$ starts to increase and converge to 1.

**The toy classifier first learns that vocab 0 is associated with the negative label and vocab 1 with the positive label under uniform attention, and then learns the attention weight $p$.**

On the other hand, suppose the classifier is initialized with $\beta_{0} = 10 > \beta_{1} = -10$, i.e. the classifier mistakens 0 as the positive token. Then $p$ will initially decrease and attend to the negative token, which is undesirable.

Finally, even though **under gradient descent with standard initialization, the gated classifier will eventually attend fully to the positive token ($p = 1$)**, we can construct a classifier with arbitrarily low loss but uniform attention.

Say, for example, $\beta_{0} = -K, \beta_{1} = 3K, p = 0.5$, then the total loss $-2K$ can be arbitrarily low. **For this classification task, low loss does not necessarily imply that the classifier will attend to key words.** 



## A Gated LSTM Binary Classifier

The above toy classifier looks interesting, but how does it relate to more complicated "real" classifiers? Here we introduce a commonly used gated-LSTM classifier and compare it to the toy classifier.

<!-- ![]({{site.baseurl}}/assets/images/model2.png) -->
<img class="post-image" src="{{site.baseurl}}/assets/images/model2.png" style="width:100%;"/>
###### An illustration of our LSTM model. Words are first embedded and then encoded by an LSTM. Next attention scores computed and used to aggergate the LSTM hidden states, before a final output transformation.


Suppose the input tokens are $[t_{1}, t_{2} \dots t_{l} \dots t_{L}]$, and we use an LSTM to encode them and obtain hidden states $[h_{1}, h_{2} \dots h_{l} \dots h_{L}]$. Then we apply an inner product with learned. vector $\alpha$ to calculate the "importance" of each hidden state:

$$a_{l} := \alpha^Th_{l}$$

Then we take the softmax over all $a_{l}$ to obtain an "attention distribution" $p_{l}$, and use it to take a weighted average $\bar{h}$ of the hidden state.

$$p_{l} := \frac{exp\{a_{l}\}}{\sum_{l'=1}^{L} exp\{a_{l'}\}}$$

$$\bar{h} := \sum_{l=1}^{L}p_{l}h_{l}$$

At last we apply a linear layer $W$ with sigmoid activation on top to produce the probability of sequence being positive.

$$Pr[Positive] := \sigma(W\bar{h}) = \sigma(\sum_{l=1}^{L}p_{l}Wh_{l})$$

We define $\beta_{l} := Wh_{l}$, which can be interpreted as "how much does the classifier consider the $l^{th}$ hidden state to be positive".

To minimize the loss, we want $\sum_{l=1}^{L}p_{l}\beta_{l}$ to be large if the label is positive, small if the label is negative. 

The form $\sum_{l=1}^{L}p_{l}\beta_{l}$ starts to resemble the scoring function of the toy classifier $c$ above.

Analogously, at the beginning of training, the attention distribution $p_{l}$ is uniform and does not change much; under uniform attention distribution, the classifier learns which word is associated with the positive label (as captured by $\beta_{l}$). Later in training the attention weights will be attracted to the "positive words". (Similar process applies to keywords that correspond to the negative words).



## Seq2Seq and a Toy Copying Task

Now let's switch to a Seq2Seq model and we study a toy task of learning the identity function (i.e. copying the input to the output, without a pointer mechanism). For example, we want our Seq2Seq architecture to learn the identity function $f$:

<center style="padding-bottom: 2%;">

$f$([1, 0, 3, 4]) = [1', 0', 3', 4']
<br/>
$f$([2, 1, 4, 3]) = [2', 1', 4', 3']
<br/>
$f$([1, 2, 3, 4]) = [1', 2', 3', 4']

</center>

While predicting the sequence [1', 0', 3', 4'] from the input sequence [1, 0, 3, 4], we expect the Seq2Seq architecture to attend to the token 1 when it is predicting the output 1', 0 when predicting the output 0'. 

However, using the intuition from before, the model needs to know that the token 1' corresponds to the input token 1 to have the "incentive" to learn to attend to token 1 while predicting 1'.

How, then, does the model learn that the output word 1' correspond to the input word 1, when the attention distribution is uniform early on during training?

For simplicity, let's assume that the average LSTM hidden state is similar to the average of the input word embeddings. 

Since the attention distribution is uniform at every step of prediction, the architecture uses the same average input word embeddings to predict the output words.

Therefore, the training process can be approximated as "using the input bag of words to predict the output bag of words", and hence the Seq2Seq architecture learns which input vocab translates to which output vocab from this bag-of-words co-occurence statistics.


We can construct a distribution that sets the bag-of-words co-occurence statistics to be non-informative, and hence the model can no longer know which input vocab translates to which output vocab, thus prevents the attention from learning.

Consider the distribution where each sequence is a permutation of 40 vocab types (as a sanity check, there will be 40! different sequences from this distribution).

Since the attention distribution is uniform at the beginning of training, a single directional Seq2Seq architecture views every sequence as the exact same bag of words, and hence frequently fails to learn this simple copying task.

In contrast, if we define the data distribution by randomly select 40 tokens from a vocab size of 60 and permute them, the architecture successfully learns this copying function every time.

This might be a bit counter-intuitive - the 2nd distribution has strictly larger support than the 1st one and hence harder to express, but the 1st one is harder to learn.

We must take the optimization trajectory into account to understand this phenomena.

<!-- ![]({{site.baseurl}}/assets/images/copying.png) -->
<img class="post-image" src="{{site.baseurl}}/assets/images/copying.png" style="width:50%;"/>
###### training on a distribution of sequences of 40 tokens chosen from a vocab of 60, verses permutations of 40 tokens. As you can see, 40 out of 60 consistenly converges, whereas the permutations frequently fail to learn.

## Limitations and Usage of Our Model

Remember that we are proposing a simplifying "model" to approximate how single head attention learns. As the old saying goes, "all models are wrong, but some are useful". Let's now retrospect what are potentially wrong with our approximation assumptions and assess the utility of our model. 

- We assumed that hidden states only reflect local input word information. Although hidden states usually encode more information about local input words, it is also informed by other words in the sequence. 
- We assumed that the attention weights are "free parameters" that can be directly optimized, while in practice it is predicted by the architecture. 
- For convenience we described the "learning to translate individual words" and "learning attention" as two separate stages in training, while in practice they are intertwined and there is not a clear boundary.
- Our model mostly reasons about early training dynamics where the attention weights are uniform; however, this is false during training time and the learned attention weights also shape subsequent training.

On the other hand, our model is useful, since:
- It explains why architectures attend to salient words by reasoning about the training dynamics, even though such a behavior is not explicitly rewarded by the loss function.
- It can predict some counter-intuitive empirical phenomena, such as the architecture sometimes fail to learn a simple identity function under a distribution of permutations.
- By formalizing the model and crisply stating the approximation assumptions, we can pinpoint which assumption is violated when the practice deviates from our theoretical prediction. This can help us develop better approximation models in the future.

