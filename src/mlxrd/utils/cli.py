"""CLI v2.0"""
import argparse

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='ML XRD CLI')
    subparsers = parser.add_subparsers(dest='command')
    
    # Init
    init_parser = subparsers.add_parser('init', help='Initialize project')
    init_parser.add_argument('--name', required=True)
    init_parser.add_argument('--path', default='.')
    
    # Build
    build_parser = subparsers.add_parser('build', help='Build dataset')
    build_parser.add_argument('--input', required=True)
    build_parser.add_argument('--output', required=True)
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--data', required=True)
    train_parser.add_argument('--model', default='xgboost')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        from .io import create_project_structure
        create_project_structure(f"{args.path}/{args.name}")
        print(f"✅ Project created: {args.name}")
    
    elif args.command == 'build':
        print(f"Building dataset from {args.input}...")
    
    elif args.command == 'train':
        print(f"Training {args.model} on {args.data}...")

if __name__ == '__main__':
    main()
