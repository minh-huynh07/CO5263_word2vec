import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np

def compare_word2vec_models():
    """General comparison script for all Word2Vec models and methods"""
    
    # Load data
    skipgram_df = pd.read_csv('train_log_skipgram.csv')
    cbow_df = pd.read_csv('train_log_cbow.csv')
    
    print("=== WORD2VEC MODELS & METHODS COMPARISON ===\n")
    
    # Find common methods between both files
    skipgram_methods = set(skipgram_df['train_method'].unique())
    cbow_methods = set(cbow_df['train_method'].unique())
    common_methods = skipgram_methods.intersection(cbow_methods)
    
    print(f"📋 Available methods:")
    print(f"  Skip-gram: {list(skipgram_methods)}")
    print(f"  CBOW: {list(cbow_methods)}")
    print(f"  Common methods to compare: {list(common_methods)}")
    print()
    
    if not common_methods:
        print("❌ No common methods found between the two files!")
        return
    
    # Create a combined dataframe for easier analysis
    all_results = []
    
    for _, row in skipgram_df.iterrows():
        if row['train_method'] in common_methods:
            all_results.append({
                'Model': 'Skip-gram',
                'Method': row['train_method'],
                'Training Time': row['train_time'],
                'Final Loss': row['final_loss'],
                'Num Pairs': row['num_pairs'],
                'Loss Per Epoch': row['loss_per_epoch']
            })
    
    for _, row in cbow_df.iterrows():
        if row['train_method'] in common_methods:
            all_results.append({
                'Model': 'CBOW',
                'Method': row['train_method'],
                'Training Time': row['train_time'],
                'Final Loss': row['final_loss'],
                'Num Pairs': row['num_pairs'],
                'Loss Per Epoch': row['loss_per_epoch']
            })
    
    results_df = pd.DataFrame(all_results)
    
    # 1. Basic comparison table
    print("📊 COMPARISON TABLE:")
    print(f"{'Model':<12} {'Method':<20} {'Time(s)':<8} {'Final Loss':<12} {'Pairs':<8}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<12} {row['Method']:<20} {row['Training Time']:<8.2f} {row['Final Loss']:<12.4f} {row['Num Pairs']:<8}")
    
    print("\n" + "="*70)
    
    # 2. Create comprehensive plots
    print("\n📈 VISUALIZATION:")
    
    num_methods = len(common_methods)
    if num_methods <= 3:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    else:
        # Adjust layout for more methods
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Training Time Comparison
    ax1 = axes[0, 0]
    models = results_df['Model'] + '-' + results_df['Method']
    times = results_df['Training Time']
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown'][:len(models)]
    bars = ax1.bar(models, times, color=colors, alpha=0.7)
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, time in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{time:.1f}s', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Final Loss Comparison
    ax2 = axes[0, 1]
    losses = results_df['Final Loss']
    bars = ax2.bar(models, losses, color=colors, alpha=0.7)
    ax2.set_ylabel('Final Loss')
    ax2.set_title('Final Loss Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, loss in zip(bars, losses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{loss:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Training Pairs Comparison
    ax3 = axes[0, 2]
    pairs = results_df['Num Pairs']
    bars = ax3.bar(models, pairs, color=colors, alpha=0.7)
    ax3.set_ylabel('Number of Training Pairs')
    ax3.set_title('Training Data Size')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, pair in zip(bars, pairs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{pair}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Loss Curves (Skip-gram)
    ax4 = axes[1, 0]
    for i, method in enumerate(common_methods):
        method_data = results_df[(results_df['Model'] == 'Skip-gram') & (results_df['Method'] == method)]
        if not method_data.empty:
            losses = ast.literal_eval(method_data.iloc[0]['Loss Per Epoch'])
            epochs = range(1, len(losses) + 1)
            ax4.plot(epochs, losses, label=method, linewidth=2, color=colors[i])
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Skip-gram Loss Curves')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Loss Curves (CBOW)
    ax5 = axes[1, 1]
    for i, method in enumerate(common_methods):
        method_data = results_df[(results_df['Model'] == 'CBOW') & (results_df['Method'] == method)]
        if not method_data.empty:
            losses = ast.literal_eval(method_data.iloc[0]['Loss Per Epoch'])
            epochs = range(1, len(losses) + 1)
            ax5.plot(epochs, losses, label=method, linewidth=2, color=colors[i])
    
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.set_title('CBOW Loss Curves')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance vs Speed
    ax6 = axes[1, 2]
    ax6.scatter(results_df['Training Time'], results_df['Final Loss'], 
               c=colors, s=100, alpha=0.7)
    
    for i, (model, method) in enumerate(zip(results_df['Model'], results_df['Method'])):
        ax6.annotate(f'{model}-{method}', 
                    (results_df['Training Time'].iloc[i], results_df['Final Loss'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=7)
    
    ax6.set_xlabel('Training Time (seconds)')
    ax6.set_ylabel('Final Loss')
    ax6.set_title('Performance vs Speed')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('word2vec_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Detailed analysis
    print("\n🔍 DETAILED ANALYSIS:")
    
    # Best performance
    best_loss_idx = results_df['Final Loss'].idxmin()
    best_model = results_df.loc[best_loss_idx, 'Model']
    best_method = results_df.loc[best_loss_idx, 'Method']
    best_loss = results_df.loc[best_loss_idx, 'Final Loss']
    print(f"• Best Performance: {best_model}-{best_method} (Loss: {best_loss:.4f})")
    
    # Fastest training
    fastest_idx = results_df['Training Time'].idxmin()
    fastest_model = results_df.loc[fastest_idx, 'Model']
    fastest_method = results_df.loc[fastest_idx, 'Method']
    fastest_time = results_df.loc[fastest_idx, 'Training Time']
    print(f"• Fastest Training: {fastest_model}-{fastest_method} (Time: {fastest_time:.2f}s)")
    
    # Method comparison
    print("\n📊 METHOD COMPARISON:")
    
    # Compare each method between CBOW and Skip-gram
    for method in common_methods:
        skipgram_data = results_df[(results_df['Model'] == 'Skip-gram') & (results_df['Method'] == method)]
        cbow_data = results_df[(results_df['Model'] == 'CBOW') & (results_df['Method'] == method)]
        
        if not skipgram_data.empty and not cbow_data.empty:
            skipgram_loss = skipgram_data.iloc[0]['Final Loss']
            cbow_loss = cbow_data.iloc[0]['Final Loss']
            skipgram_time = skipgram_data.iloc[0]['Training Time']
            cbow_time = cbow_data.iloc[0]['Training Time']
            
            print(f"• {method.replace('_', ' ').title()}: CBOW vs Skip-gram")
            print(f"  - Loss: {cbow_loss:.4f} vs {skipgram_loss:.4f} ({'CBOW better' if cbow_loss < skipgram_loss else 'Skip-gram better'})")
            print(f"  - Speed: {cbow_time:.2f}s vs {skipgram_time:.2f}s (CBOW {skipgram_time/cbow_time:.2f}x faster)")
    
    # Model comparison
    print("\n📊 MODEL COMPARISON:")
    
    # Compare each model across methods
    models = ['Skip-gram', 'CBOW']
    
    for model in models:
        model_data = results_df[results_df['Model'] == model]
        if not model_data.empty:
            best_method_idx = model_data['Final Loss'].idxmin()
            best_method = model_data.loc[best_method_idx, 'Method']
            best_loss = model_data.loc[best_method_idx, 'Final Loss']
            fastest_method_idx = model_data['Training Time'].idxmin()
            fastest_method = model_data.loc[fastest_method_idx, 'Method']
            fastest_time = model_data.loc[fastest_method_idx, 'Training Time']
            
            print(f"• {model}:")
            print(f"  - Best Method: {best_method} (Loss: {best_loss:.4f})")
            print(f"  - Fastest Method: {fastest_method} (Time: {fastest_time:.2f}s)")
    
    # Overall recommendations
    print(f"\n🏆 RECOMMENDATIONS:")
    print(f"• Best Overall: {best_model}-{best_method}")
    print(f"• Fastest: {fastest_model}-{fastest_method}")
    
    # Best for each criterion
    best_loss_time_ratio_idx = (results_df['Final Loss'] / results_df['Training Time']).idxmin()
    best_ratio_model = results_df.loc[best_loss_time_ratio_idx, 'Model']
    best_ratio_method = results_df.loc[best_loss_time_ratio_idx, 'Method']
    print(f"• Best Loss/Time Ratio: {best_ratio_model}-{best_ratio_method}")
    
    print(f"\n💾 Plot saved to: word2vec_comparison.png")

if __name__ == "__main__":
    compare_word2vec_models() 