from ..models.huggingface_model import WhiteBoxModelBase
from ..models.model_base import ModelBase
from .attacker_base import AttackerBase
from ..seed.seed_random import SeedRandom
from ..mutation.unaligned_token_gradient import MutationUnalignedTokenGradient
from ..selector.HiddenLayerLossSelector import HiddenLayerLossSelector
from ..metrics.Evaluator_MultiAgentJudge import EvaluatorMultiAgentJudge
from ..datasets.jailbreak_datasets import JailbreakDataset
from ..datasets.instance import Instance

import os
import logging
from typing import Optional

class POEF(AttackerBase):
    def __init__(
        self,
        attack_model: WhiteBoxModelBase,
        target_model: ModelBase,
        eval_model: ModelBase,
        unaligned_model: ModelBase,
        jailbreak_datasets: JailbreakDataset,
        jailbreak_prompt_length: int = 20,
        num_turb_sample: int = 512,
        batchsize: Optional[int] = None,
        max_num_iter: int = 500,
        is_universal: bool = False
    ):
        """
        Initialize the POEF attacker.

        :param WhiteBoxModelBase attack_model: Model used to compute gradient variations and select optimal mutations based on loss.
        :param ModelBase target_model: Model used to generate target responses.
        :param ModelBase unaligned_model: Model used to compute loss.
        :param JailbreakDataset jailbreak_datasets: Dataset for the attack.
        :param int jailbreak_prompt_length: Number of tokens in the jailbreak prompt. Defaults to 20.
        :param int num_turb_sample: Number of mutant samples generated per instance. Defaults to 512.
        :param Optional[int] batchsize: Batch size for computing loss during the selection of optimal mutant samples.
            If encountering OOM errors, consider reducing this value. Defaults to None, which is set to the same as num_turb_sample.
        :param int max_num_iter: Maximum number of iterations. Will exit early if all samples are successfully attacked.
            Defaults to 500.
        :param bool is_universal: Experimental feature. Optimize a shared jailbreak prompt for all instances. Defaults to False.
        """

        super().__init__(attack_model, target_model, None, jailbreak_datasets)

        if batchsize is None:
            batchsize = num_turb_sample

        self.seeder = SeedRandom(seeds_max_length=jailbreak_prompt_length, posible_tokens=['ok '])
        self.mutator = MutationUnalignedTokenGradient(attack_model, unaligned_model, num_turb_sample=num_turb_sample, is_universal=is_universal)
        self.selector = HiddenLayerLossSelector(attack_model, unaligned_model, batch_size=batchsize, is_universal=is_universal)
        self.evaluator = EvaluatorMultiAgentJudge(eval_model)
        self.max_num_iter = max_num_iter

    def single_attack(self, instance: Instance):
        dataset = self.jailbreak_datasets    # FIXME
        self.jailbreak_datasets = JailbreakDataset([instance])
        self.attack()
        ans = self.jailbreak_datasets
        self.jailbreak_datasets = dataset
        return ans

    def attack(self):
        logging.info("Jailbreak started!")
        try:
            for instance in self.jailbreak_datasets:
                seed = self.seeder.new_seeds()[0]     # FIXME: The design of the seed part needs to be reconsidered
                if instance.jailbreak_prompt is None:
                    instance.jailbreak_prompt = f'{{query}} {seed}'
            
            breaked_dataset = JailbreakDataset([])
            unbreaked_dataset = self.jailbreak_datasets
            for epoch in range(self.max_num_iter):
                logging.info(f"Current POEF epoch: {epoch}/{self.max_num_iter}")
                unbreaked_dataset = self.mutator(unbreaked_dataset)
                logging.info(f"Mutation: {len(unbreaked_dataset)} new instances generated.")
                unbreaked_dataset = self.selector.select(unbreaked_dataset)
                logging.info(f"Selection: {len(unbreaked_dataset)} instances selected.")
                for instance in unbreaked_dataset:
                    prompt = instance.jailbreak_prompt.replace('{query}', instance.query)
                    logging.info(f'Generation: input=`{prompt}`')
                    instance.target_responses = [self.target_model.generate(prompt)]
                    logging.info(f'Generation: Output=`{instance.target_responses}`')
                self.evaluator(unbreaked_dataset)
                self.jailbreak_datasets = JailbreakDataset.merge([unbreaked_dataset, breaked_dataset])

                # check
                cnt_attack_success = 0
                breaked_dataset = JailbreakDataset([])
                unbreaked_dataset = JailbreakDataset([])
                for instance in self.jailbreak_datasets:
                    if instance.eval_results[-1]:
                        cnt_attack_success += 1
                        breaked_dataset.add(instance)
                    else:
                        unbreaked_dataset.add(instance)
                logging.info(f"Successfully attacked: {cnt_attack_success}/{len(self.jailbreak_datasets)}")
                if os.environ.get('CHECKPOINT_DIR') is not None:
                    checkpoint_dir = os.environ.get('CHECKPOINT_DIR')
                    self.jailbreak_datasets.save_to_jsonl(f'{checkpoint_dir}/poef_{epoch}.jsonl')
                if cnt_attack_success == len(self.jailbreak_datasets):
                    break   # all instances is successfully attacked
        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")

        self.log_results(cnt_attack_success)
        logging.info("Jailbreak finished!")
