import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import styled from 'styled-components'


const SideBarDiv = styled.div`
    height: 100vh;
    overflow: auto;
    position: fixed;
    z-index: 1;
    display: block;
    width: 400px;
    background: white;
    right: 0px;
`

export function Sidebar() {
    return (
        <SideBarDiv>

        </SideBarDiv>
    )
}