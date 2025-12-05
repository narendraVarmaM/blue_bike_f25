"""
Setup Feature View for Blue Bikes Inference Pipeline

This script creates a feature view from the feature group to enable
batch inference. Run this once after the feature group has been populated.
"""

import hopsworks

import src.bluebikes_config as config

print("=" * 70)
print("CREATING BLUE BIKES FEATURE VIEW")
print("=" * 70)

# Connect to Hopsworks
print("\n[1/3] Connecting to Hopsworks...")
project = hopsworks.login(
    project=config.HOPSWORKS_PROJECT_NAME,
    api_key_value=config.HOPSWORKS_API_KEY
)
print(f"✓ Connected to project: {config.HOPSWORKS_PROJECT_NAME}")

# Get feature store
print("\n[2/3] Connecting to feature store...")
feature_store = project.get_feature_store()
print("✓ Connected to feature store")

# Get the feature group
print(f"\n[3/3] Creating feature view...")
print(f"  Feature group: {config.FEATURE_GROUP_NAME} v{config.FEATURE_GROUP_VERSION}")
print(f"  Feature view: {config.FEATURE_VIEW_NAME} v{config.FEATURE_VIEW_VERSION}")

try:
    feature_group = feature_store.get_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION
    )

    # Select all features from the feature group
    query = feature_group.select_all()

    # Create or get feature view
    feature_view = feature_store.get_or_create_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION,
        description="Blue Bikes Boston - Hourly demand features for inference",
        query=query,
    )

    print(f"✓ Feature view created: {feature_view.name} v{feature_view.version}")

except Exception as e:
    print(f"✗ Error: {e}")
    raise

print("\n" + "=" * 70)
print("SETUP COMPLETED")
print("=" * 70)
print("\nYou can now run the inference pipeline:")
print("  python -m pipelines.bluebikes_inference_pipeline")
