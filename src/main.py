#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:
Version:
Autor: dreamy-xay
Date: 2024-10-22 11:16:02
LastEditors: dreamy-xay
LastEditTime: 2024-11-16 14:22:24
"""
from engine.mtable import MTableArgParser, MTableTrainer, MTableValidator, MTablePredictor


def main():

    args = MTableArgParser().parse()

    if args.mode == "train":
        trainer = MTableTrainer(args)
        trainer.run()
    elif args.mode == "val":
        validator = MTableValidator(args)
        validator.run()
    elif args.mode == "predict":
        predictor = MTablePredictor(args)
        predictor.run()
    else:
        raise ValueError("mode must be 'train' or 'val' or 'predict'")


if __name__ == "__main__":
    # Main function
    main()
