-- Returns 0 rows
-- NOTE: Needs to be checked for RxNorm Extension, too
with
    strength as (
        select
            drug_concept_id,
            concept_class_id,
            ingredient_concept_id,
            array[
                ingredient_concept_id,
                amount_value,
                amount_unit_concept_id,
                numerator_value,
                numerator_unit_concept_id,
                denominator_value,
                denominator_unit_concept_id
            ] as ingredient_strength
        from drug_strength d
        join
            concept c
            on d.drug_concept_id = c.concept_id
            and c.concept_class_id
            in ('Branded Drug Comp', 'Branded Drug', 'Clinical Drug')
            and c.vocabulary_id = 'RxNorm'
            and c.standard_concept = 'S'
    ),
    strength_agg as (
        select
            drug_concept_id,
            concept_class_id,
            array_agg(ingredient_strength order by ingredient_concept_id) as strength
        from strength
        group by drug_concept_id, concept_class_id
        having count(ingredient_concept_id) > 1
    ),
    family as (
        select
            bd.drug_concept_id,
            bc.drug_concept_id,
            cd.drug_concept_id,
            bd.strength = bc.strength as matches_component,
            bd.strength = cd.strength as matches_clinical
        from strength_agg bd
        join
            concept_relationship r1
            on bd.drug_concept_id = r1.concept_id_1
            and r1.invalid_reason is null
        -- r1.relationship_id = 'Consists of'
        join
            strength_agg bc
            on bc.concept_class_id = 'Branded Drug Comp'
            and bc.drug_concept_id = r1.concept_id_2
        join
            concept_relationship r2
            on bd.drug_concept_id = r2.concept_id_1
            and r1.invalid_reason is null
        -- r1.relationship_id = 'Tradename of'
        join
            strength_agg cd
            on cd.concept_class_id = 'Clinical Drug'
            and cd.drug_concept_id = r2.concept_id_2
        where bd.concept_class_id = 'Branded Drug'
    )
select *
from family
where not (matches_component and matches_clinical)
;
