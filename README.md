

# Towards Safe Synthetic Image Generation On the Web: A Multimodal Robust NSFW Defense and Million Scale Dataset

## Abstract
In the past years, we have witnessed the remarkable success of Text-to-Image (T2I) models and their widespread use on the web. Extensive research in making T2I models produce hyper-realistic images has led to new concerns, such as generating Not-Safe-For-Work (NSFW) web content and polluting the online society with the misuse of such technologies. To this end, defensive mechanisms such as NSFW and post-hoc security filters are implemented in T2I models to mitigate the misuse of T2I models and develop a safe online ecosystem for web users. However, recent work unveiled how these methods can easily fail to prevent misuse. In particular, careful adversarial attacks on text and image modalities can easily outplay defensive measures.

Moreover, there is no robust million-scale multimodal NSFW dataset with both prompt and image pairs with adversarial examples. In this work, we propose a large-scale prompt and image dataset, generated using open-source diffusion models. Also, we develop a multimodal classification model to distinguish safe and NSFW text and images, which has robustness against adversarial attacks, and directly alleviates the current challenges. Our extensive experimental results show that our model shows good performance against existing SOTA NSFW detection methods in terms of accuracy and recall, and drastically reduced the Attack Success Rate (ASR) in multimodal adversarial attack scenarios.

The code and the restricted dataset are available at the following GitHub repository: [https://github.com/shahidmuneer/multimodal-nsfw-defense](https://github.com/shahidmuneer/multimodal-nsfw-defense).

---

## Repository Structure

This repository contains the following:

- **Training and Validation Scripts**:
  - `train.py`: Script for training the multimodal classification model.
  - `validate.py`: Script for evaluating the model performance.

---

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch 2.0+
- Additional dependencies listed in `requirements.txt`

Install the dependencies using:
```bash
pip install torch pandas diffusers transformers
```

---

## Training
To train the model, use the following command:
```bash
python train.py
```

---

## Validation
To validate the model, use the following command:
```bash
python validate.py
```

---



---

## License
This repository is licensed under [MIT License](LICENSE).

---

## Contact
For questions or feedback, feel free to contact:
- **Shahid Muneer**: [shahidmuneer@g.skku.edu](mailto:shahidmuneer@g.skku.edu)

We welcome contributions and collaborations!
