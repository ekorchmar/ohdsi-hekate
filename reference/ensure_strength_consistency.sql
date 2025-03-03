-- Returns 0 rows
-- NOTE: Needs to be checked for RxNorm Extension, too
with
strength as (
    select
        d.drug_concept_id,
        c.concept_class_id,
        c.ingredient_concept_id,
        array[
            d.ingredient_concept_id,
            d.amount_value,
            d.amount_unit_concept_id,
            d.numerator_value,
            d.numerator_unit_concept_id,
            d.denominator_value,
            d.denominator_unit_concept_id
        ] as ingredient_strength
    from drug_strength as d
    inner join
        concept as c
        on
            d.drug_concept_id = c.concept_id
            and c.concept_class_id
            in ('Branded Drug Comp', 'Branded Drug', 'Clinical Drug')
            and c.vocabulary_id = 'RxNorm'
            and c.standard_concept = 'S'
),

strength_agg as (
    select
        drug_concept_id,
        concept_class_id,
        array_agg(
            ingredient_strength
            order by ingredient_concept_id
        ) as strength
    from strength
    group by drug_concept_id, concept_class_id
    having count(ingredient_concept_id) > 1
),

family as (
    select
        bd.drug_concept_id as branded_drug_concept_id,
        bc.drug_concept_id as branded_comp_concept_id,
        cd.drug_concept_id as clinical_drug_concept_id,
        bd.strength = bc.strength as matches_component,
        bd.strength = cd.strength as matches_clinical
    from strength_agg as bd
    inner join
        concept_relationship as r1
        on
            bd.drug_concept_id = r1.concept_id_1
            and r1.invalid_reason is null
    -- r1.relationship_id = 'Consists of'
    inner join
        strength_agg as bc
        on
            bc.concept_class_id = 'Branded Drug Comp'
            and r1.concept_id_2 = bc.drug_concept_id
    inner join
        concept_relationship as r2
        on
            bd.drug_concept_id = r2.concept_id_1
            and r1.invalid_reason is null
    -- r1.relationship_id = 'Tradename of'
    inner join
        strength_agg as cd
        on
            cd.concept_class_id = 'Clinical Drug'
            and r2.concept_id_2 = cd.drug_concept_id
    where bd.concept_class_id = 'Branded Drug'
)

select *
from family
where not (matches_component and matches_clinical);
