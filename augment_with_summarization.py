import argparse
import logging
from pprint import pformat

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from transformers.utils import PaddingStrategy

from src.data import DoceeForInference

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("CLI args")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Generative model which will be used for summarization"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed which controls randomness during text generation (if any)"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to a csv file containing the dataset to be augmented."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to which the augmented dataset will be saved."
    )
    parser.add_argument(
        "--device",
        type=int,
        required=True,
        help="Device to use during inference. -1 defaults to CPU, while >=0 specifies a GPU number."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, logging level will be set to info."
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=20,
        help="Minimum summarization length. Defaults to 20."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum summarization length. Defaults to 100."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams to use for beam search. Defaults to 4."
             "Setting this value to 1 means no beam search."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size to use during inference"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=4,
        help="Number of workers to use for loading the data. Defaults to 4."
    )
    parser.add_argument(
        "--low_resource_cutoff",
        type=int,
        required=False,
        default=None,
        help="If provided, will only augment examples from classes which have no more examples than the "
             "low resource cutoff."
    )
    return parser


def main():
    args = get_parser().parse_args()
    logging.root.handlers = []
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # log info only on main process
        level=logging.INFO if args.verbose else logging.WARNING,
        handlers=[
            logging.StreamHandler(),
        ],
    )

    df = pd.read_csv(args.input_path)
    logger.info(f"Loaded {len(df)} examples.")

    # df has "id" column

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        padding=PaddingStrategy.MAX_LENGTH,
        use_fast=True
    )
    summarizer = pipeline(
        "summarization",
        model=args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        device=args.device,
        framework="pt"
    )

    summary_df = df.loc[:, :]  # retain all columns (including the id)

    if args.low_resource_cutoff:
        logger.info(f"Low resource cutoff set to {args.low_resource_cutoff}."
                    f"Augmenting only classes with no more than "
                    f"{args.low_resource_cutoff} examples.")
        # filter based on class count
        from src.utils import low_resource_slice
        low_resource_classes, summary_df = low_resource_slice(
            summary_df,
            args.low_resource_cutoff,
            return_classes=True
        )
        logger.info(f"Low resource classes: {pformat(low_resource_classes)}")
        logger.info(f"{len(summary_df) = }")

    # after low-resource slice, ids are still retained

    dataset = DoceeForInference(summary_df)

    summary_df.loc[:, "text"] = [
        out[0]["summary_text"] for out in tqdm(summarizer(
            dataset,
            min_length=args.min_length,
            max_length=args.max_length,
            num_beams=args.num_beams,
            truncation=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        ), desc=f"Inference loop", total=len(dataset))

        # early_stopping :: bool
        #   if set to True, then generation will finish as soon as all beams generate EOS token
        #   tbh it doesnt make sense to ever not set this to true
        #
        # no_repeat_ngram_size :: int
        #   ngrams of that size will not be repeated during text generation
        #   careful about that -> e.g. if "New York" is present in the article,
        #   and we set no_repeat_ngram_size to 2, then "New York" will never repeat
        #
        # num_return_sequences :: int
        #   set this to return multiple sequences at once. TODO : check how does this impact memory usage
        #
        # do_sampling :: bool
        #   if set to true, w_t is sampled from P(w | w_{1:(t-1)}). In other words, text
        #   generation is no longer deterministic
        #   -> generation depends on the random seed. TODO : implement a function for random seeding
        #
        # temperature :: float
        #   temperature controls the sharpness of the probability distribution generated by softmax
        #   softmax(x;T) = exp(x/T) / sum_i exp(x_i / T)
        #   T -> 0 -- distribution decays into a dirac impulse at the most probable point (equal to greedy decoding)
        #   T -> inf -- distribution decays into an uniform distribution
        #   T < 1 -> distribution is sharper  (bias towards tokens with higher probability)
        #   T > 1 -> distribution is smoother
        #
        # top_k :: int
        #   if set to >0, applies top-k sampling. In each time step, only the top k most probable tokens
        #   are considered, and probability mass is redistributed amongst those tokens before sampling
        #   -> the problem with top_k is that it doesn't dynamically adapt the number of words to sample
        #
        # top_p :: float
        #   if set to >0, applies top-p (nucleus) sampling. A mimimum size set of most probable tokens
        #   is chosen, such that the probability mass of the set exceeds the top_p parameter.
        #   this is good because the size of the set is dynamic and depends on the concrete distribution
        #   of next token
        #
        # penalty_alpha -> alpha for contrastive search.
        #   contrastive search is supposed to be SOTA, but after reading the paper, it seems that:
        #       contrastive search only works better than top-p when used with the model
        #       which was trained with the addition of contrastive loss
        #     -> we can resolve this issue by training BART on CNN/DM using contrastive loss,
        #           but that frankly seems like an overkill
        # TODO - implement a CLI interface for different generating strategy:
        #   beam_search
        #   greedy
        #   top-k
        #   top-p
        #   contrastive_search
        #
        # TODO - implement generation of multiple sequences at once
        # TODO - figure out what is the best way to save the generated summaries
        # TODO - add source document ID for each summary

    ]

    summary_df.loc[:, "source_doc_id"] = summary_df.loc[:, "id"]
    df_to_save = pd.concat((df.loc[:, :], summary_df))

    # reset index
    df_to_save.drop(columns=["id"], inplace=True)
    df_to_save.reset_index(names="id", inplace=True)
    # since unsummarized examples come first, their ids will correctly
    #   be set in accordance to source_doc_id
    # unsummarized examples will have source_doc_id set to NaN

    logging.info(f"Length of concatenated dataset: {len(df_to_save)}")
    logging.info(pformat(df_to_save.head()))
    logging.info(f"Columns: {df_to_save.columns}")

    df_to_save.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    main()
