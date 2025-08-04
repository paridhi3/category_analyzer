# Configure columns with text wrapping
        column_config = {
            "File Name": st.column_config.TextColumn(
                "File Name",
                width="medium",
                help="Name of the case study file"
            ),
            "Category": st.column_config.TextColumn(
                "Category",
                width="small"
            ),
            "Domain": st.column_config.TextColumn(
                "Domain",
                width="medium"
            ),
            "Client Name": st.column_config.TextColumn(
                "Client Name",
                width="medium"
            ),
            "Technologies Used": st.column_config.TextColumn(
                "Technologies Used",
                width="large",
                help="Technologies and tools used in the project"
            ),
        }
        
        st.dataframe(
            df, 
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )
