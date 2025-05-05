import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from jinja2 import Template


class ReportGenerator:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.report_dir = self.results_dir / 'report'
        self.report_dir.mkdir(exist_ok=True)
    
    def generate_html_report(self, results: Dict[str, Any]):
        """Generate a comprehensive HTML report."""
        # Load HTML template
        template = self._get_report_template()
        
        # Prepare data for the report
        report_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_performance': self._format_model_performance(results),
            'clinical_metrics': self._format_clinical_metrics(results),
            'network_analysis': self._format_network_analysis(results),
            'regional_analysis': self._format_regional_analysis(results),
            'temporal_analysis': self._format_temporal_analysis(results),
            'group_differences': self._format_group_differences(results),
            'reliability_analysis': self._format_reliability_analysis(results)
        }
        
        # Generate interactive visualizations
        self._generate_interactive_plots(results)
        
        # Render HTML report
        html_content = template.render(**report_data)
        
        # Save report
        with open(self.report_dir / 'evaluation_report.html', 'w') as f:
            f.write(html_content)
    
    def _get_report_template(self) -> Template:
        """Get the HTML template for the report."""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Autism Detection Model Evaluation Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                .metric-card {
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin: 10px;
                    border-radius: 5px;
                }
                .plot-container {
                    margin: 20px 0;
                }
            </style>
        </head>
        <body>
            <div class="container mt-5">
                <h1>Autism Detection Model Evaluation Report</h1>
                <p class="text-muted">Generated on: {{ timestamp }}</p>
                
                <section class="mt-5">
                    <h2>Model Performance Summary</h2>
                    <div class="row">
                        {% for metric in model_performance %}
                        <div class="col-md-3">
                            <div class="metric-card">
                                <h5>{{ metric.name }}</h5>
                                <h3>{{ metric.value }}</h3>
                                <p class="text-muted">{{ metric.description }}</p>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    <div class="plot-container">
                        <div id="roc_curve"></div>
                        <div id="pr_curve"></div>
                    </div>
                </section>
                
                <section class="mt-5">
                    <h2>Clinical Metrics</h2>
                    <div class="row">
                        {% for metric in clinical_metrics %}
                        <div class="col-md-4">
                            <div class="metric-card">
                                <h5>{{ metric.name }}</h5>
                                <h3>{{ metric.value }}</h3>
                                <p>{{ metric.interpretation }}</p>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </section>
                
                <section class="mt-5">
                    <h2>Network Analysis</h2>
                    <div id="network_metrics"></div>
                    {{ network_analysis | safe }}
                </section>
                
                <section class="mt-5">
                    <h2>Regional Analysis</h2>
                    <div id="regional_importance"></div>
                    {{ regional_analysis | safe }}
                </section>
                
                <section class="mt-5">
                    <h2>Temporal Analysis</h2>
                    <div id="temporal_patterns"></div>
                    {{ temporal_analysis | safe }}
                </section>
                
                <section class="mt-5">
                    <h2>Group Differences</h2>
                    <div id="group_differences"></div>
                    {{ group_differences | safe }}
                </section>
                
                <section class="mt-5">
                    <h2>Reliability Analysis</h2>
                    <div id="reliability_metrics"></div>
                    {{ reliability_analysis | safe }}
                </section>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            <script src="plots.js"></script>
        </body>
        </html>
        """
        return Template(template_str)
    
    def _format_model_performance(self, results: Dict[str, Any]) -> list:
        """Format model performance metrics."""
        return [
            {
                'name': 'ROC AUC',
                'value': f"{results['roc_auc']:.3f}",
                'description': 'Area under the ROC curve'
            },
            {
                'name': 'PR AUC',
                'value': f"{results['pr_auc']:.3f}",
                'description': 'Area under the Precision-Recall curve'
            },
            {
                'name': 'Accuracy',
                'value': f"{results['clinical_metrics']['balanced_accuracy']:.3f}",
                'description': 'Balanced accuracy'
            }
        ]
    
    def _format_clinical_metrics(self, results: Dict[str, Any]) -> list:
        """Format clinical metrics with interpretations."""
        metrics = results['clinical_metrics']
        interpretations = {
            'sensitivity': 'Proportion of actual positive cases correctly identified',
            'specificity': 'Proportion of actual negative cases correctly identified',
            'ppv': 'Probability that subjects with positive test truly have the condition',
            'npv': 'Probability that subjects with negative test truly don\'t have the condition',
            'diagnostic_odds_ratio': 'Ratio of odds of positivity in subjects with disease to odds in subjects without disease'
        }
        
        return [
            {
                'name': k.replace('_', ' ').title(),
                'value': f"{v:.3f}",
                'interpretation': interpretations.get(k, '')
            }
            for k, v in metrics.items()
        ]
    
    def _format_network_analysis(self, results: Dict[str, Any]) -> str:
        """Format network analysis results."""
        metrics = results['network_metrics']
        return f"""
        <div class="row">
            <div class="col-md-6">
                <h4>Key Findings:</h4>
                <ul>
                    <li>Global Efficiency: {metrics['global_efficiency']:.3f}</li>
                    <li>Clustering Coefficient: {metrics['clustering_coefficient']:.3f}</li>
                    <li>Characteristic Path Length: {metrics['characteristic_path_length']:.3f}</li>
                    <li>Modularity: {metrics['modularity']:.3f}</li>
                </ul>
            </div>
        </div>
        """
    
    def _format_regional_analysis(self, results: Dict[str, Any]) -> str:
        """Format regional analysis results."""
        metrics = results['regional_metrics']
        return f"""
        <div class="row">
            <div class="col-12">
                <p>Analysis of {len(metrics['attention_strength'])} brain regions</p>
            </div>
        </div>
        """
    
    def _format_temporal_analysis(self, results: Dict[str, Any]) -> str:
        """Format temporal analysis results."""
        metrics = results['temporal_metrics']
        return f"""
        <div class="row">
            <div class="col-12">
                <p>Temporal patterns analyzed across {len(metrics['temporal_variability'])} time points</p>
            </div>
        </div>
        """
    
    def _format_group_differences(self, results: Dict[str, Any]) -> str:
        """Format group difference results."""
        diff = results['group_differences']
        return f"""
        <div class="row">
            <div class="col-md-6">
                <h4>Statistical Analysis:</h4>
                <ul>
                    <li>Effect Size (Cohen's d): {diff['cohens_d']:.3f}</li>
                    <li>T-statistic: {diff['t_statistic']:.3f}</li>
                    <li>P-value: {diff['p_value']:.3e}</li>
                </ul>
            </div>
        </div>
        """
    
    def _format_reliability_analysis(self, results: Dict[str, Any]) -> str:
        """Format reliability analysis results."""
        metrics = results['reliability_metrics']
        return f"""
        <div class="row">
            <div class="col-md-6">
                <h4>Reliability Metrics:</h4>
                <ul>
                    <li>Prediction Consistency: {metrics['prediction_consistency']:.3f}</li>
                    <li>Pattern Consistency: {metrics['pattern_consistency']:.3f}</li>
                    <li>ICC Score: {metrics['icc_score']:.3f}</li>
                </ul>
            </div>
        </div>
        """
    
    def _generate_interactive_plots(self, results: Dict[str, Any]):
        """Generate interactive plots using plotly."""
        plots_js = []
        
        # ROC Curve
        roc = go.Figure()
        roc.add_trace(go.Scatter(
            x=results['roc_curve']['fpr'],
            y=results['roc_curve']['tpr'],
            name=f"ROC (AUC = {results['roc_auc']:.3f})"
        ))
        roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            line=dict(dash='dash'),
            name='Random'
        ))
        roc.update_layout(title='ROC Curve')
        plots_js.append(f"Plotly.newPlot('roc_curve', {roc.to_json()})")
        
        # Network Metrics
        network = go.Figure(data=[
            go.Bar(
                x=list(results['network_metrics'].keys()),
                y=list(results['network_metrics'].values())
            )
        ])
        network.update_layout(title='Network Metrics')
        plots_js.append(f"Plotly.newPlot('network_metrics', {network.to_json()})")
        
        # Save plots.js
        with open(self.report_dir / 'plots.js', 'w') as f:
            f.write('document.addEventListener("DOMContentLoaded", function() {\n')
            f.write('\n'.join(plots_js))
            f.write('\n});')
