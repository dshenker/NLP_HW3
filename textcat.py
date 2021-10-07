#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path
import matplotlib.pyplot as plt
from probs import LanguageModel, num_tokens, read_trigrams
import numpy as np
log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model1",
        type=Path,
        help="path to the trained model for generic",
    )
    parser.add_argument(
        "model2",
        type=Path,
        help="path to the trained model for spam",
    )
    parser.add_argument(
        "prior",
        type=float,
        help="prior probability of generic",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0
    for (x, y, z) in read_trigrams(file, lm.vocab):
        prob = lm.prob(x, y, z)  # p(z | xy)
        log_prob += math.log(prob) 
    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    log.info("Testing...")
    lm1 = LanguageModel.load(args.model1)
    lm2 = LanguageModel.load(args.model2)
    prior = args.prior

    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # But we'd like to print a value in bits: so we convert
    # log base e to log base 2 at print time, by dividing by log(2).

    log.info("Per-file log-probabilities:")
    
    more_likely_gen = []
    more_likely_spam = []
    all_file_lengths = []
    all_file_labels = []
    file_error_list = []
    for file in args.test_files:
        file_name = str(file)
        file_name_slashes = file_name.split('/')
        file_name_important = file_name_slashes[-1]
        file_name_pieces = file_name_important.split('.')
        file_cat = file_name_pieces[0]
        file_length = file_name_pieces[1]
        
        log_prob_gen: float = file_log_prob(file, lm1)
        log_prob_spam: float = file_log_prob(file, lm2)        
        log_prob_gen += math.log(prior)
        log_prob_spam += math.log((1 - prior))
        if (log_prob_gen > log_prob_spam):
            more_likely_gen.append(file)
            print("gen" + '\t' + str(file))
        else:
            more_likely_spam.append(file)
            print("spam" + '\t' + str(file))

        all_file_lengths.append(int(file_length))
        if (file_cat == "gen") & (log_prob_spam > log_prob_gen):
            #file_error_list.append(int(file_length))
            all_file_labels.append(0)
        elif (file_cat == "spam") & (log_prob_gen > log_prob_spam):
            #file_error_list.append(int(file_length))
            all_file_labels.append(0)
        else:
            all_file_labels.append(1)
    num_gen = len(more_likely_gen)
    num_spam = len(more_likely_spam)
    num_tot = num_gen + num_spam
    pct_gen = num_gen / num_tot * 100
    pct_spam = num_spam / num_tot * 100
    print(str(num_gen) + " files were more likely gen (" + str(pct_gen) + "%)")
    print(str(num_spam) + " files were more likely spam (" + str(pct_spam) + "%)")
    
    #print(len(file_error_list) / len(all_file_lengths))
    print(np.mean(all_file_labels))
    print(np.corrcoef(all_file_lengths, all_file_labels))
   # plt.hist(file_error_list, bins = 50, alpha = 0.5,label = "errors")
   # plt.hist(all_file_lengths, bins = 50, alpha = 0.5, label = "number files")
   # plt.xlabel("file length")
   # plt.ylabel("number of errors")
   # plt.title("Relationship Between Length of File and Error Rate")
   # plt.show()

    #plt.hist(all_file_lengths)
    #plt.xlabel("file length")
    #plt.ylabel("number of files")
    #plt.title("Distribution of File Lengths")
    #plt.show()


if __name__ == "__main__":
    main()
