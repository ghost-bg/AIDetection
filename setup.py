#!/usr/bin/env python3
from datasets import load_dataset
from openai import OpenAI
import yaml


ds = load_dataset("toughdata/quora-question-answer-dataset")
