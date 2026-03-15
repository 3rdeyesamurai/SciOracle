import argparse
import sys
import yaml
from agents.research_agent import AutonomousResearchAgent
from memory.database import TheoremDatabase

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_cli():
    """Command Line Parser for standalone discovery application."""
    parser = argparse.ArgumentParser(description="SciOracle v2 Autonomous Logic Engine")
    
    subparsers = parser.add_subparsers(dest='command', help='Select operations')
    
    # 1. train the model
    train_parser = subparsers.add_parser('train', help='Train the internal neural embedding layers')
    
    # 2. explore mathematical identities
    explore_parser = subparsers.add_parser('explore', help='Explore an initial symbolic equation')
    explore_parser.add_argument('equation', type=str, help='Expression to start search from')
    
    # 3. query the theorem database
    db_parser = subparsers.add_parser('query', help='Read proven knowledge graph database points')
    
    # 4. run autonomous discovery mode
    auto_parser = subparsers.add_parser('auto', help='Loop the comprehensive search -> train agent')
    auto_parser.add_argument('--iters', type=int, default=10, help='Loops for agent')

    args = parser.parse_args()
    
    config = load_config()

    if args.command == 'train':
        print("Starting neural component training script...")
        agent = AutonomousResearchAgent()
        agent.train_loop()
    elif args.command == 'explore':
        print(f"Executing manual search tree for {args.equation}")
        agent = AutonomousResearchAgent()
        import sympy as sp
        expr = sp.sympify(args.equation)
        path = agent.mcts.search(expr)
        print("Path Discovered:")
        for step in path:
            print(f" -> {step['operation']}: {step['result']}")
    elif args.command == 'query':
        print("Stored Theorems from DB:")
        db = TheoremDatabase(db_path=config['database']['db_path'])
        theorems = db.get_all_theorems()
        for count, t in enumerate(theorems):
            print(f"  {count + 1} | Initially: {t[1]} => Finally: {t[2]}")
    elif args.command == 'auto':
        print(f"Running automated scientific agent pipeline for {args.iters} iterations.")
        agent = AutonomousResearchAgent()
        agent.discover_loop(args.iters)
    else:
        parser.print_help()

if __name__ == '__main__':
    run_cli()
