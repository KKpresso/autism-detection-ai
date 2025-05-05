import sys
import os
from pathlib import Path
import torch
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.models.graph_model import AutismClassifier
from backend.data.preprocessing import load_and_preprocess_fmri
from backend.evaluation.model_evaluation import ModelEvaluator
from backend.evaluation.generate_report import ReportGenerator


def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / 'evaluation_results' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting comprehensive model evaluation...")
    print(f"Results will be saved to: {output_dir}")
    
    # Load data
    print("\n1. Loading and preprocessing data...")
    data_dir = project_root / 'data'
    dataset = load_and_preprocess_fmri(data_dir)
    
    # Initialize model
    print("\n2. Initializing model...")
    num_regions = dataset[0][0].shape[1]  # Get number of brain regions from data
    model = AutismClassifier(
        num_regions=num_regions,
        hidden_dim=64,
        embedding_dim=32,
        num_heads=4
    )
    
    # Load best model weights if available
    model_path = project_root / 'models' / 'best_model.pt'
    if model_path.exists():
        print("Loading pre-trained model...")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, output_dir=output_dir)
    
    # Perform cross-validation
    print("\n3. Performing cross-validation...")
    cv_results = evaluator.cross_validate(
        dataset=dataset,
        n_splits=5,
        batch_size=32,
        num_epochs=50
    )
    
    # Create data loader for final evaluation
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )
    
    # Analyze brain regions
    print("\n4. Analyzing brain regions...")
    region_names = [f"Region_{i}" for i in range(num_regions)]  # Replace with actual region names
    atlas_path = data_dir / 'atlas.nii.gz'  # Replace with actual atlas path
    region_importance = evaluator.analyze_brain_regions(
        data_loader=data_loader,
        region_names=region_names,
        atlas_path=str(atlas_path)
    )
    
    # Generate comprehensive performance report
    print("\n5. Generating performance report...")
    performance_results = evaluator.generate_performance_report(data_loader)
    
    # Generate HTML report
    print("\n6. Generating HTML report...")
    report_generator = ReportGenerator(output_dir)
    report_generator.generate_html_report(performance_results)
    
    # Print summary
    print("\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("\nKey Metrics:")
    print(f"Mean validation accuracy: {sum(cv_results['val_acc']) / len(cv_results['val_acc']):.4f}")
    print(f"Mean validation AUC-ROC: {sum(cv_results['val_auc']) / len(cv_results['val_auc']):.4f}")
    print(f"Diagnostic Odds Ratio: {performance_results['clinical_metrics']['diagnostic_odds_ratio']:.4f}")
    print("\nTop 5 most important brain regions:")
    for region, importance in list(region_importance.items())[:5]:
        print(f"{region}: {importance:.4f}")
    
    print(f"\nDetailed HTML report available at: {output_dir}/report/evaluation_report.html")


if __name__ == "__main__":
    main()
