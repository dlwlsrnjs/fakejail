# Clustered contextual BAI replay (v3)

This evaluates success discovery over the ten observed target settings per person; it is not a 360-arm stochastic BAI claim.

## asr

Observed oracle@10: 98.0%

| policy | B=1 | B=2 | B=3 | B=5 | B=10 | gap@3 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| random | 75.6% | 85.8% | 90.2% | 93.6% | 98.0% | 7.8 |
| surrogate_static | 70.9% | 80.8% | 85.8% | 94.6% | 98.0% | 12.2 |
| nested_best_router_static | 78.8% | 89.8% | 92.2% | 94.4% | 98.0% | 5.8 |
| context_cluster_static | 81.2% | 91.4% | 93.4% | 94.8% | 98.0% | 4.6 |
| triple_clst_adapted | 81.6% | 90.6% | 93.2% | 95.4% | 98.0% | 4.8 |
| ccb_adaptive | 81.2% | 90.2% | 92.8% | 95.4% | 98.0% | 5.2 |
| budget_aware_clustered_bai | 81.2% | 89.8% | 93.0% | 95.2% | 98.0% | 5.0 |

## retained

Observed oracle@10: 70.1%

| policy | B=1 | B=2 | B=3 | B=5 | B=10 | gap@3 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| random | 18.4% | 32.9% | 41.7% | 54.3% | 70.1% | 28.3 |
| surrogate_static | 22.2% | 31.9% | 41.7% | 55.5% | 70.1% | 28.3 |
| nested_best_router_static | 22.0% | 34.1% | 40.5% | 55.1% | 70.1% | 29.5 |
| context_cluster_static | 22.0% | 34.1% | 40.5% | 55.1% | 70.1% | 29.5 |
| triple_clst_adapted | 21.2% | 34.1% | 42.1% | 54.5% | 70.1% | 27.9 |
| ccb_adaptive | 22.0% | 33.7% | 42.7% | 54.1% | 70.1% | 27.3 |
| budget_aware_clustered_bai | 22.8% | 33.7% | 43.3% | 54.9% | 70.1% | 26.7 |

## strict

Observed oracle@10: 26.3%

| policy | B=1 | B=2 | B=3 | B=5 | B=10 | gap@3 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| random | 4.8% | 8.8% | 12.2% | 16.2% | 26.3% | 14.2 |
| surrogate_static | 4.0% | 7.6% | 10.0% | 15.4% | 26.3% | 16.4 |
| nested_best_router_static | 5.0% | 9.2% | 12.6% | 17.0% | 26.3% | 13.8 |
| context_cluster_static | 5.0% | 9.2% | 12.6% | 17.0% | 26.3% | 13.8 |
| triple_clst_adapted | 5.8% | 9.2% | 11.0% | 17.2% | 26.3% | 15.4 |
| ccb_adaptive | 5.0% | 8.6% | 11.8% | 17.4% | 26.3% | 14.6 |
| budget_aware_clustered_bai | 5.8% | 8.4% | 12.6% | 17.0% | 26.3% | 13.8 |

## Chosen fold configurations

```json
{
  "asr": [
    {
      "fold": 0,
      "train_people": 395,
      "test_people": 106,
      "base_model": "joint_interaction_router",
      "chosen": {
        "blend": 0.5,
        "cluster_weight": 0.5,
        "kernel_weight": 0.0,
        "novelty_weight": 0.15
      },
      "inner_score": 0.9205766160514532,
      "controller_by_budget": {
        "1": "context_cluster_static",
        "2": "ccb_adaptive",
        "3": "ccb_adaptive",
        "4": "ccb_adaptive",
        "5": "ccb_adaptive",
        "6": "triple_clst_adapted",
        "7": "ccb_adaptive",
        "8": "ccb_adaptive",
        "9": "context_cluster_static",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 395,
        "cluster_sizes": {
          "0": 42,
          "1": 17,
          "2": 14,
          "3": 26,
          "4": 20,
          "5": 7,
          "6": 2,
          "7": 29,
          "8": 10,
          "9": 2,
          "10": 30,
          "11": 16,
          "12": 33,
          "13": 29,
          "14": 17,
          "15": 101
        },
        "global_rate": 0.731586940015186,
        "soft_assignment_mean_entropy": 2.667463133152047
      }
    },
    {
      "fold": 1,
      "train_people": 398,
      "test_people": 103,
      "base_model": "pc2_static_language_router",
      "chosen": {
        "blend": 0.5,
        "cluster_weight": 0.5,
        "kernel_weight": 0.0,
        "novelty_weight": 0.0
      },
      "inner_score": 0.9077111839911818,
      "controller_by_budget": {
        "1": "context_cluster_static",
        "2": "context_cluster_static",
        "3": "triple_clst_adapted",
        "4": "triple_clst_adapted",
        "5": "triple_clst_adapted",
        "6": "triple_clst_adapted",
        "7": "context_cluster_static",
        "8": "context_cluster_static",
        "9": "context_cluster_static",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 398,
        "cluster_sizes": {
          "0": 31,
          "1": 27,
          "2": 17,
          "3": 50,
          "4": 31,
          "5": 14,
          "6": 26,
          "7": 32,
          "8": 4,
          "9": 21,
          "10": 17,
          "11": 28,
          "12": 20,
          "13": 19,
          "14": 34,
          "15": 27
        },
        "global_rate": 0.7248178849535293,
        "soft_assignment_mean_entropy": 2.703233015804562
      }
    },
    {
      "fold": 2,
      "train_people": 407,
      "test_people": 94,
      "base_model": "formula_method_prior",
      "chosen": {
        "blend": 0.5,
        "cluster_weight": 0.5,
        "kernel_weight": 0.0,
        "novelty_weight": 0.0
      },
      "inner_score": 0.9099499815263862,
      "controller_by_budget": {
        "1": "triple_clst_adapted",
        "2": "context_cluster_static",
        "3": "context_cluster_static",
        "4": "triple_clst_adapted",
        "5": "triple_clst_adapted",
        "6": "ccb_adaptive",
        "7": "ccb_adaptive",
        "8": "context_cluster_static",
        "9": "context_cluster_static",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 407,
        "cluster_sizes": {
          "0": 19,
          "1": 67,
          "2": 26,
          "3": 30,
          "4": 17,
          "5": 4,
          "6": 12,
          "7": 3,
          "8": 23,
          "9": 39,
          "10": 16,
          "11": 45,
          "12": 10,
          "13": 33,
          "14": 10,
          "15": 53
        },
        "global_rate": 0.7306558585114222,
        "soft_assignment_mean_entropy": 2.6838819539214627
      }
    },
    {
      "fold": 3,
      "train_people": 421,
      "test_people": 80,
      "base_model": "intent_technique_router",
      "chosen": {
        "blend": 0.5,
        "cluster_weight": 0.5,
        "kernel_weight": 0.0,
        "novelty_weight": 0.0
      },
      "inner_score": 0.9204436002520048,
      "controller_by_budget": {
        "1": "context_cluster_static",
        "2": "ccb_adaptive",
        "3": "ccb_adaptive",
        "4": "context_cluster_static",
        "5": "context_cluster_static",
        "6": "context_cluster_static",
        "7": "context_cluster_static",
        "8": "context_cluster_static",
        "9": "context_cluster_static",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 421,
        "cluster_sizes": {
          "0": 11,
          "1": 46,
          "2": 62,
          "3": 16,
          "4": 27,
          "5": 27,
          "6": 36,
          "7": 24,
          "8": 38,
          "9": 17,
          "10": 33,
          "11": 25,
          "12": 5,
          "13": 29,
          "14": 16,
          "15": 9
        },
        "global_rate": 0.7429351697933982,
        "soft_assignment_mean_entropy": 2.702141662583628
      }
    },
    {
      "fold": 4,
      "train_people": 383,
      "test_people": 118,
      "base_model": "joint_interaction_router",
      "chosen": {
        "blend": 0.5,
        "cluster_weight": 1.5,
        "kernel_weight": 0.0,
        "novelty_weight": 0.0
      },
      "inner_score": 0.9212654829229888,
      "controller_by_budget": {
        "1": "context_cluster_static",
        "2": "ccb_adaptive",
        "3": "ccb_adaptive",
        "4": "ccb_adaptive",
        "5": "ccb_adaptive",
        "6": "triple_clst_adapted",
        "7": "ccb_adaptive",
        "8": "ccb_adaptive",
        "9": "context_cluster_static",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 383,
        "cluster_sizes": {
          "0": 2,
          "1": 35,
          "2": 22,
          "3": 33,
          "4": 21,
          "5": 37,
          "6": 21,
          "7": 32,
          "8": 16,
          "9": 4,
          "10": 31,
          "11": 26,
          "12": 36,
          "13": 11,
          "14": 30,
          "15": 26
        },
        "global_rate": 0.7307491516575306,
        "soft_assignment_mean_entropy": 2.6996487674698764
      }
    }
  ],
  "retained": [
    {
      "fold": 0,
      "train_people": 395,
      "test_people": 106,
      "base_model": "formula_method_prior",
      "chosen": {
        "blend": 0.0,
        "cluster_weight": 1.5,
        "kernel_weight": 0.0,
        "novelty_weight": 0.15
      },
      "inner_score": 0.4234574314909022,
      "controller_by_budget": {
        "1": "triple_clst_adapted",
        "2": "triple_clst_adapted",
        "3": "ccb_adaptive",
        "4": "ccb_adaptive",
        "5": "ccb_adaptive",
        "6": "ccb_adaptive",
        "7": "ccb_adaptive",
        "8": "ccb_adaptive",
        "9": "context_cluster_static",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 395,
        "cluster_sizes": {
          "0": 42,
          "1": 17,
          "2": 14,
          "3": 26,
          "4": 20,
          "5": 7,
          "6": 2,
          "7": 29,
          "8": 10,
          "9": 2,
          "10": 30,
          "11": 16,
          "12": 33,
          "13": 29,
          "14": 17,
          "15": 101
        },
        "global_rate": 0.2058972412047583,
        "soft_assignment_mean_entropy": 2.667463133152047
      }
    },
    {
      "fold": 1,
      "train_people": 398,
      "test_people": 103,
      "base_model": "entity_language_router",
      "chosen": {
        "blend": 0.0,
        "cluster_weight": 0.5,
        "kernel_weight": 0.75,
        "novelty_weight": 0.15
      },
      "inner_score": 0.4169594650912778,
      "controller_by_budget": {
        "1": "context_cluster_static",
        "2": "ccb_adaptive",
        "3": "triple_clst_adapted",
        "4": "triple_clst_adapted",
        "5": "context_cluster_static",
        "6": "triple_clst_adapted",
        "7": "ccb_adaptive",
        "8": "triple_clst_adapted",
        "9": "ccb_adaptive",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 398,
        "cluster_sizes": {
          "0": 31,
          "1": 27,
          "2": 17,
          "3": 50,
          "4": 31,
          "5": 14,
          "6": 26,
          "7": 32,
          "8": 4,
          "9": 21,
          "10": 17,
          "11": 28,
          "12": 20,
          "13": 19,
          "14": 34,
          "15": 27
        },
        "global_rate": 0.2068575734740015,
        "soft_assignment_mean_entropy": 2.703233015804562
      }
    },
    {
      "fold": 2,
      "train_people": 407,
      "test_people": 94,
      "base_model": "entity_language_router",
      "chosen": {
        "blend": 0.0,
        "cluster_weight": 1.5,
        "kernel_weight": 0.0,
        "novelty_weight": 0.0
      },
      "inner_score": 0.41071272443888335,
      "controller_by_budget": {
        "1": "context_cluster_static",
        "2": "ccb_adaptive",
        "3": "triple_clst_adapted",
        "4": "triple_clst_adapted",
        "5": "context_cluster_static",
        "6": "triple_clst_adapted",
        "7": "context_cluster_static",
        "8": "triple_clst_adapted",
        "9": "ccb_adaptive",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 407,
        "cluster_sizes": {
          "0": 19,
          "1": 67,
          "2": 26,
          "3": 30,
          "4": 17,
          "5": 4,
          "6": 12,
          "7": 3,
          "8": 23,
          "9": 39,
          "10": 16,
          "11": 45,
          "12": 10,
          "13": 33,
          "14": 10,
          "15": 53
        },
        "global_rate": 0.20793416850896584,
        "soft_assignment_mean_entropy": 2.6838819539214627
      }
    },
    {
      "fold": 3,
      "train_people": 421,
      "test_people": 80,
      "base_model": "entity_language_router",
      "chosen": {
        "blend": 0.0,
        "cluster_weight": 0.5,
        "kernel_weight": 0.0,
        "novelty_weight": 0.15
      },
      "inner_score": 0.42822824336444504,
      "controller_by_budget": {
        "1": "context_cluster_static",
        "2": "triple_clst_adapted",
        "3": "triple_clst_adapted",
        "4": "triple_clst_adapted",
        "5": "context_cluster_static",
        "6": "triple_clst_adapted",
        "7": "ccb_adaptive",
        "8": "context_cluster_static",
        "9": "ccb_adaptive",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 421,
        "cluster_sizes": {
          "0": 11,
          "1": 46,
          "2": 62,
          "3": 16,
          "4": 27,
          "5": 27,
          "6": 36,
          "7": 24,
          "8": 38,
          "9": 17,
          "10": 33,
          "11": 25,
          "12": 5,
          "13": 29,
          "14": 16,
          "15": 9
        },
        "global_rate": 0.214794585609119,
        "soft_assignment_mean_entropy": 2.702141662583628
      }
    },
    {
      "fold": 4,
      "train_people": 383,
      "test_people": 118,
      "base_model": "entity_language_router",
      "chosen": {
        "blend": 0.0,
        "cluster_weight": 1.5,
        "kernel_weight": 0.0,
        "novelty_weight": 0.0
      },
      "inner_score": 0.4121191434796334,
      "controller_by_budget": {
        "1": "context_cluster_static",
        "2": "ccb_adaptive",
        "3": "triple_clst_adapted",
        "4": "triple_clst_adapted",
        "5": "context_cluster_static",
        "6": "ccb_adaptive",
        "7": "context_cluster_static",
        "8": "triple_clst_adapted",
        "9": "ccb_adaptive",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 383,
        "cluster_sizes": {
          "0": 2,
          "1": 35,
          "2": 22,
          "3": 33,
          "4": 21,
          "5": 37,
          "6": 21,
          "7": 32,
          "8": 16,
          "9": 4,
          "10": 31,
          "11": 26,
          "12": 36,
          "13": 11,
          "14": 30,
          "15": 26
        },
        "global_rate": 0.2011224223440355,
        "soft_assignment_mean_entropy": 2.6996487674698764
      }
    }
  ],
  "strict": [
    {
      "fold": 0,
      "train_people": 395,
      "test_people": 106,
      "base_model": "formula_method_prior",
      "chosen": {
        "blend": 0.0,
        "cluster_weight": 0.5,
        "kernel_weight": 0.0,
        "novelty_weight": 0.0
      },
      "inner_score": 0.12069666452511912,
      "controller_by_budget": {
        "1": "triple_clst_adapted",
        "2": "context_cluster_static",
        "3": "context_cluster_static",
        "4": "context_cluster_static",
        "5": "triple_clst_adapted",
        "6": "triple_clst_adapted",
        "7": "triple_clst_adapted",
        "8": "ccb_adaptive",
        "9": "context_cluster_static",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 395,
        "cluster_sizes": {
          "0": 42,
          "1": 17,
          "2": 14,
          "3": 26,
          "4": 20,
          "5": 7,
          "6": 2,
          "7": 29,
          "8": 10,
          "9": 2,
          "10": 30,
          "11": 16,
          "12": 33,
          "13": 29,
          "14": 17,
          "15": 101
        },
        "global_rate": 0.03758542141230068,
        "soft_assignment_mean_entropy": 2.667463133152047
      }
    },
    {
      "fold": 1,
      "train_people": 398,
      "test_people": 103,
      "base_model": "formula_method_prior",
      "chosen": {
        "blend": 0.0,
        "cluster_weight": 0.5,
        "kernel_weight": 0.0,
        "novelty_weight": 0.0
      },
      "inner_score": 0.12146603620491397,
      "controller_by_budget": {
        "1": "triple_clst_adapted",
        "2": "context_cluster_static",
        "3": "context_cluster_static",
        "4": "context_cluster_static",
        "5": "ccb_adaptive",
        "6": "triple_clst_adapted",
        "7": "triple_clst_adapted",
        "8": "ccb_adaptive",
        "9": "context_cluster_static",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 398,
        "cluster_sizes": {
          "0": 31,
          "1": 27,
          "2": 17,
          "3": 50,
          "4": 31,
          "5": 14,
          "6": 26,
          "7": 32,
          "8": 4,
          "9": 21,
          "10": 17,
          "11": 28,
          "12": 20,
          "13": 19,
          "14": 34,
          "15": 27
        },
        "global_rate": 0.0405676965586536,
        "soft_assignment_mean_entropy": 2.703233015804562
      }
    },
    {
      "fold": 2,
      "train_people": 407,
      "test_people": 94,
      "base_model": "formula_method_prior",
      "chosen": {
        "blend": 0.0,
        "cluster_weight": 1.5,
        "kernel_weight": 0.0,
        "novelty_weight": 0.15
      },
      "inner_score": 0.11461961238011556,
      "controller_by_budget": {
        "1": "triple_clst_adapted",
        "2": "context_cluster_static",
        "3": "context_cluster_static",
        "4": "context_cluster_static",
        "5": "ccb_adaptive",
        "6": "ccb_adaptive",
        "7": "ccb_adaptive",
        "8": "ccb_adaptive",
        "9": "context_cluster_static",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 407,
        "cluster_sizes": {
          "0": 19,
          "1": 67,
          "2": 26,
          "3": 30,
          "4": 17,
          "5": 4,
          "6": 12,
          "7": 3,
          "8": 23,
          "9": 39,
          "10": 16,
          "11": 45,
          "12": 10,
          "13": 33,
          "14": 10,
          "15": 53
        },
        "global_rate": 0.038442643085237045,
        "soft_assignment_mean_entropy": 2.6838819539214627
      }
    },
    {
      "fold": 3,
      "train_people": 421,
      "test_people": 80,
      "base_model": "formula_method_prior",
      "chosen": {
        "blend": 0.0,
        "cluster_weight": 1.5,
        "kernel_weight": 0.0,
        "novelty_weight": 0.15
      },
      "inner_score": 0.12162759110351981,
      "controller_by_budget": {
        "1": "triple_clst_adapted",
        "2": "triple_clst_adapted",
        "3": "context_cluster_static",
        "4": "context_cluster_static",
        "5": "ccb_adaptive",
        "6": "ccb_adaptive",
        "7": "ccb_adaptive",
        "8": "ccb_adaptive",
        "9": "ccb_adaptive",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 421,
        "cluster_sizes": {
          "0": 11,
          "1": 46,
          "2": 62,
          "3": 16,
          "4": 27,
          "5": 27,
          "6": 36,
          "7": 24,
          "8": 38,
          "9": 17,
          "10": 33,
          "11": 25,
          "12": 5,
          "13": 29,
          "14": 16,
          "15": 9
        },
        "global_rate": 0.040489194965566376,
        "soft_assignment_mean_entropy": 2.702141662583628
      }
    },
    {
      "fold": 4,
      "train_people": 383,
      "test_people": 118,
      "base_model": "formula_method_prior",
      "chosen": {
        "blend": 0.0,
        "cluster_weight": 1.5,
        "kernel_weight": 0.0,
        "novelty_weight": 0.15
      },
      "inner_score": 0.10744326906962151,
      "controller_by_budget": {
        "1": "triple_clst_adapted",
        "2": "triple_clst_adapted",
        "3": "context_cluster_static",
        "4": "context_cluster_static",
        "5": "ccb_adaptive",
        "6": "ccb_adaptive",
        "7": "ccb_adaptive",
        "8": "ccb_adaptive",
        "9": "context_cluster_static",
        "10": "context_cluster_static"
      },
      "context_clusters": {
        "clusters": 16,
        "train_people": 383,
        "cluster_sizes": {
          "0": 2,
          "1": 35,
          "2": 22,
          "3": 33,
          "4": 21,
          "5": 37,
          "6": 21,
          "7": 32,
          "8": 16,
          "9": 4,
          "10": 31,
          "11": 26,
          "12": 36,
          "13": 11,
          "14": 30,
          "15": 26
        },
        "global_rate": 0.03693552597233098,
        "soft_assignment_mean_entropy": 2.6996487674698764
      }
    }
  ]
}
```
