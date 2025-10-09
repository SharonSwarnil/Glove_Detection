# Part 2: Reasoning-Based Questions - Answers

---

## Q1: Choosing the Right Approach**

I would use **object detection** to identify whether a product is missing its label on the assembly line. Detection allows the model to both localize and classify regions of interest — in this case, detecting whether a label exists or not. A simple classification model would only indicate if a label is present in the overall image, without showing *where* it is located, which limits practical use for automation.  
Segmentation could be used if pixel-level precision is needed, but detection provides a more efficient balance between accuracy and speed for this task.  
If detection doesn’t perform well due to subtle visual differences between labeled and unlabeled products, my fallback would be a **segmentation-based model** (like Mask R-CNN or YOLO-Seg) to analyze exact label regions at the pixel level.

---

## Q2: Debugging a Poorly Performing Model**

If my trained model performs poorly on new factory images, I would start by checking for **data drift or domain shift** — for example, differences in lighting, camera angle, or background between training and deployment data.  
Next, I would visualize several **failure cases** by plotting predictions vs. ground truth to understand where the model fails (misclassification, missed detection, etc.).  
I’d also verify the **quality and balance** of my dataset, ensuring both classes are well represented and labels are accurate. Additionally, I’d inspect preprocessing or augmentation steps that might have unintentionally distorted critical visual details.  
Finally, I’d fine-tune the model on a **small, curated validation set** that matches real factory conditions to test if performance improves, helping isolate whether the problem is with data, labels, or model generalization.

---

## Q3: Accuracy vs Real Risk**

Accuracy isn’t the right metric in this case because it doesn’t reflect the **actual operational risk**. Even with 98% accuracy, missing 1 out of 10 defective products could cause major safety or compliance issues.  
In such scenarios, I’d focus on **Recall (Sensitivity)** for the defective class — prioritizing catching all defects, even if a few false positives occur. Precision, Recall, F1-Score, and the **Confusion Matrix** provide a better understanding of real-world reliability.  
For high-risk industrial applications, **minimizing false negatives** (missed defects) is far more critical than achieving a high overall accuracy score.

---

## **Q4: Annotation Edge Cases**

Blurry or partially visible objects should be **included selectively** in the dataset. If the real-world production environment sometimes captures unclear images, keeping some of them helps the model become more robust under realistic conditions.  
However, if the blur or partial visibility makes it impossible to label the object correctly, such samples should be excluded to avoid confusing the model.  
The trade-off lies between **realism and label certainty** — retaining moderately unclear images improves generalization, while removing highly ambiguous ones maintains dataset quality.  
A balanced dataset that mirrors deployment conditions but maintains clear labeling standards ensures the best model performance.

---

**Prepared by:** *Sharon Swarnil Choudhary*   
**Project:** Gloved vs. Ungloved Hand Detection System (Safety Compliance)
