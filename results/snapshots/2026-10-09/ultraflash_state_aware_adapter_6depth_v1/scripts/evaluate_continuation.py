"""Reuse the unchanged Val24 evaluator for continuation checkpoints only."""
import evaluate

evaluate.STEPS = (1500, 2000, 2500, 3000)

if __name__ == '__main__':
    evaluate.main()
