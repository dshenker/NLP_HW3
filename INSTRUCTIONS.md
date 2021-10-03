Hooray, let's kick it with some smoothed language models.

## Downloading the Assignment Materials

We assume that you've made a local copy of
<http://www.cs.jhu.edu/~jason/465/hw-lm/> (for example, by downloading
and unpacking the zipfile there) and that you're currently in the
`code/` subdirectory.

## Environments and Miniconda

You probably also want to install Miniconda, a minimal tool
for managing and reproducing environments. It does the hard
work of installing packages like NumPy that use faster,
vectorized math compared to the standard Python int and float
data types.

Miniconda (and its big sibling Anaconda) are all the rage in
NLP and deep learning. Install it following your platform-
specific instructions from here:

<https://conda.io/projects/conda/en/latest/user-guide/install/index.html>

One you've installed it, you can create an "environment" that
matches the one on the autograder, so you instantly know whether
your code will work there or not.

    conda env create -f environment.yml

Now that you've cloned our environment (made it available for your use)
from the .yml specification file, you can "activate" it.

    conda activate hw-lm

If this worked, then your prompt should be prefixed by the 
environment name, like this:

    (hw-lm) arya@ugradx:~/hw-lm/code$

This means that third-party packages like PyTorch are now
available for you to "import" in your Python scripts. You
are also, for sure, using the same Python version as we are.

----------

## QUESTION 1.

We provide a script `./build_vocab.py` for you to build a vocabulary
from some corpus.  Type `./build_vocab.py --help` to see
documentation.  Once you've familiarized yourself with the arguments,
try running it like this:

    ./build_vocab.py ../data/gen_spam/train/{gen,spam} --threshold 3 --output vocab-genspam.txt 

This creates `vocab-genspam.txt`, which you can look at: it's just a set of word types.

Once you've built a vocab file, you can use it to build one or more
smoothed language models.  If you are *comparing* two models, both
models should use the *same* vocab file, to make the probabilities
comparable (as explained in the homework handout).

We also provide a script `./train_lm.py` for you to build a smoothed
language model from a vocab file and a corpus.  (The code for actually
training and using models is in the `probs.py` module, which you will
extend later.)

Type `./train_lm.py --help` to see documentation.  Once you've
familiarized yourself with the arguments, try running it like this:

    ./train_lm.py vocab-genspam.txt add_lambda --lambda 1.0 ../data/gen_spam/train/gen 

Here `add_lambda` is the type of smoothing, and `--lambda` specifies
the hyperparameter λ=1.0.  While the documentation mentions additional
hyperparameters like `--l2_regularization`, they are not used by the
`add_lambda` smoothing technique, so specifying them will have no
effect on it.

Since the above command line doesn't specify an `--output` file to
save the model in, the script just makes up a long filename (ending in
`.model`) that mentions the choice of hyperparameters.  You may
sometimes want to use shorter filenames, or specific filenames that
are required by the submission instructions that we'll post on Piazza.

The file
`corpus=gen~vocab=vocab-genspam.txt~smoother=add_lambda~lambda=1.0.model`.
now contains a
[pickled](https://docs.python.org/3/library/pickle.html) copy of a
trained Python `LanguageModel` object.  The object contains everything
you need to *use* the language model, including the type of language
model, the trained parameters, and a copy of the vocabulary.  Other
scripts can just load the model object from the file and query it to
get information like $p(z \mid xy)$ by calling its methods. They don't
need to know how the model works internally or how it was trained.

You can now use your trained models to assign probabilities to new
corpora using `./fileprob.py`.  Type `./fileprob.py --help` to see
documentation.  Once you've familiarized yourself with the arguments,
try running the script like this:

    ./fileprob.py [mymodel] ../data/gen_spam/dev/gen/*

where `[mymodel]` refers to the long filename above.  (You may not
have to type it all: try typing the start and hitting Tab, or type
`*.model` if it's the only model matching that pattern.)

*Note:* It may be convenient to use symbolic links (shortcuts) to
avoid typing long filenames or directory names.  For example,

    ln -sr corpus=gen~vocab=vocab-genspam.txt~smoother=add_lambda~lambda=1.0.model gen

will make `gen` be a shortcut for the long model filename, and

	ln -sr ../data/speech/train sptrain 

will make `sptrain` be a shortcut to that directory, so that `sptrain/switchboard` is now a shortcut to `../data/speech/train/switchboard`.

----------

## QUESTIONS 2-3.

Copy `fileprob.py` to `textcat.py`.

Modify `textcat.py` so that it does text categorization. `textcat.py`
should have almost the same command-line API as `./fileprob.py`,
except it should take *two* models instad of just one.

You could train your language models with lines like

    ./train_lm.py vocab-genspam.txt add_lambda --lambda 1.0 gen --output gen.model
    ./train_lm.py vocab-genspam.txt add_lambda --lambda 1.0 spam --output spam.model

which saves the trained models in a file but prints no output.  You should then
be able to categorize the development corpus files in question 3 like this:

    ./textcat.py gen.model spam.model 0.7 ../data/gen_spam/dev/{gen,spam}/*

Note that `LanguageModel` objects have a `vocab` attribute.  You
should do a sanity check in `textcat.py` that both language models
loaded for text categorization have the same vocabulary.  If not,
`raise` an exception, or alternatively, just print an error message (`log.error`) 
and halt (`sys.exit(1)`).

(It's generally wise to include sanity checks in your code that will
immediately catch problems, so that you don't have to track down
mysterious behavior.  The `assert` statement is used to check
statements that should be correct if your code is *internally*
correct.  Once your code is correct, these assertions should *never*
fail.  Some people even turn assertion-checking off in the final
version, for speed.  But even correct code may encounter conditions
beyond its control; for those cases, you should `raise` an exception
to warn the caller that the code couldn't do what it was asked to do,
typically because the arguments were bad or the required resources
were unavailable.)

----------

## QUESTION 5.

You want to support the `add_lambda_backoff` argument to
`train_lm.py`.  This makes use of `BackoffAddLambdaLanguageModel`
class in `probs.py`.  You will have to implement the `prob()` method
in that class.

Make sure that for any $z$, you have $\sum_z p(z \mid xy) = 1$, where
$z$ ranges over the whole vocabulary including OOV and EOS.

As you are only adding a new model, the behavior of your old models such
as `AddLambdaLanguageModel` should not change.

----------------------------------------------------------------------

## QUESTION 6.

Now add the `sample()` method to `probs.py`.  Did your good
object-oriented programming principles suggest the best place to do
this?

To make `trigram_randsent.py`, start by copying `fileprob.py`.  As the
handout indicates, the graders should be able to call the script like
this:

    ./trigram_randsent.py [mymodel] 10 --max_length 20

to get 10 samples of length at most 20.

----------------------------------------------------------------------

## QUESTION 7.

You want to support the `log_linear` argument to `train_lm.py`.
This makes use of `EmbeddingLogLinearLanguageModel` in `probs.py`.
Complete that class.

For part (b), you'll need to complete the `train()` method in that class.

For part (d), you want to support `log_linear_improved`.  This makes
use of `ImprovedLogLinearLanguageModel`, which you should complete as
you see fit.  It is a subclass of the LOGLIN model, so you can inherit or
override methods as you like.

As you are only adding new models, the behavior of your old models
should not change.

### Using vector/matrix operations (crucial for speed!)

Training the log-linear model on `en.1K` can be done with simple "for" loops and
2D array representation of matrices.  However, you're encouraged to use
PyTorch's tensor operations, as discussed in the handout.  This will reduce 
training time and might simplify your code.

*TA's note:* "My original implementation took 22 hours per epoch. Careful
vectorization of certain operations, leveraging PyTorch, brought that
runtime down to 13 minutes per epoch."

Make sure to use the `torch.logsumexp` method for computing the log-denominator
in the log-probability.

### Improve the SGD training loop (optional)

The reading handout has a section with this title.

To recover Algorithm 1 (convergent SGD), you can use a modified
optimizer that we provide for you in `SGD_convergent.py`:

    from SGD_convergent import ConvergentSGD
    optimizer = ConvergentSGD(self.parameters(), gamma0=gamma0, lambda_=2*C/N)

To break the epoch model as suggested in the "Shuffling" subsection, 
check out the method `draw_trigrams_forever` in `probs.py`.

For mini-batching, you could modify either `read_trigrams` or `draw_trigrams_forever`.

----------------------------------------------------------------------

## QUESTION 9 (EXTRA CREDIT)

You can use the same language models as before, without changing
`probs.py` or `train_lm.py`.

In this question, however, you're back to using only one language
model as in `fileprob` (not two as in `textcat`).  So, initialize
`speechrec.py` to a copy of `fileprob.py`, and then edit it.

Modify `speechrec.py` so that, instead of evaluating the prior
probability of the entire test file, it separately evaluates the prior
probability of each candidate transcription in the file.  It can then
select the transcription with the highest *posterior* probability and
report its error rate, as required.

The `read_trigrams` function in `probs.py` is no longer useful, since a
speech dev or test file has a special format.  You don't want to
iterate over all the trigrams in such a file.  You may want to make an
"outer loop" utility function that iterates over the candidate
transcriptions in a given speech dev or test file, along with an
"inner loop" utility function that iterates over the trigrams in a
given candidate transcription.

(The outer loop is specialized to the speechrec format, so it probably
belongs in `speechrec.py`.  The inner loop is similar to
`read_trigrams` and might be more generally useful, so it probably
belongs in `probs.py`.)

----------------------------------------------------------------------

## CREDITS

A version of this Python port for an earlier version of this
assignment was kindly provided by Eric Perlman, a previous student in
the NLP class.  Thanks to Prof. Jason Baldridge at U. of Texas for
updating the code and these instructions when he borrowed that
assignment.  They were subsequently modified for later versions of
the assignment by Xuchen Yao, Mozhi Zhang, Chu-Cheng Lin, Arya McCarthy,
Brian Lu, and Jason Eisner.
